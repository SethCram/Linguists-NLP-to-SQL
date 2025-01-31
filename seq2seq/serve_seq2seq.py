# Set up logging
import sys
import logging

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.WARNING,
)
logger = logging.getLogger(__name__)

from typing import Optional, Dict
from dataclasses import dataclass, field
from pydantic import BaseModel
import os
from contextlib import nullcontext
from transformers.hf_argparser import HfArgumentParser
from transformers.models.auto import AutoConfig, AutoTokenizer, AutoModelForSeq2SeqLM
from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from sqlite3 import Connection, connect, OperationalError
from seq2seq.utils.pipeline import Text2SQLGenerationPipeline, Text2SQLInput, get_schema
from seq2seq.utils.picard_model_wrapper import PicardArguments, PicardLauncher, with_picard
from seq2seq.utils.dataset import DataTrainingArguments
import shutil
import subprocess
from uvicorn import run

@dataclass
class BackendArguments:
    """
    Arguments pertaining to model serving.
    """

    model_path: str = field(
        default="tscholak/cxmefzzi",
        metadata={"help": "Path to pretrained model"},
    )
    cache_dir: Optional[str] = field(
        default="/tmp",
        metadata={"help": "Where to cache pretrained models and data"},
    )
    db_path: str = field(
        default="database",
        metadata={"help": "Where to to find the sqlite files"},
    )
    sql_path: str = field(
        default="sql",
        metadata={"help": "Where to to find the sqlite files"},
    )
    host: str = field(default="0.0.0.0", metadata={"help": "Bind socket to this host"})
    port: int = field(default=8000, metadata={"help": "Bind socket to this port"})
    device: int = field(
        default=0,
        metadata={
            "help": "Device ordinal for CPU/GPU supports. Setting this to -1 will leverage CPU. A non-negative value will run the model on the corresponding CUDA device id."
        },
    )


def main():
    # See all possible arguments by passing the --help flag to this program.
    parser = HfArgumentParser((PicardArguments, BackendArguments, DataTrainingArguments))
    picard_args: PicardArguments
    backend_args: BackendArguments
    data_training_args: DataTrainingArguments
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        picard_args, backend_args, data_training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        picard_args, backend_args, data_training_args = parser.parse_args_into_dataclasses()

    # Initialize config
    config = AutoConfig.from_pretrained(
        backend_args.model_path,
        cache_dir=backend_args.cache_dir,
        max_length=data_training_args.max_target_length,
        num_beams=data_training_args.num_beams,
        num_beam_groups=data_training_args.num_beam_groups,
        diversity_penalty=data_training_args.diversity_penalty,
    )

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        backend_args.model_path,
        cache_dir=backend_args.cache_dir,
        use_fast=True,
    )
    
    #region Helper Functs
        
    def store_file(file, destination: str, byte_mode: bool = False):
        """Stores file in destination through writing in byte mode. 

        Args:
            file (_type_): _description_
            destination (str): file path to copy file to
            byte_mode (bool): Whether to copy the file in using bytes or not (for binary)
        Raises:
            HTTPException: _Raises error 500 if other problem occures._
            HTTPException: _Raises error 400 if file already exists._
        """
        
        #if file exists, return it
        if( os.path.exists(destination)):
            raise HTTPException(status_code=400, detail=f"File already exists at {destination}. Rename your uploaded file or delete the pre-existing one(s).")
        
        operational_mode = "w"
        
        if(byte_mode):
            operational_mode += "b"
        
        try:
            with open(destination, operational_mode) as file_obj:
                shutil.copyfileobj(file.file, file_obj)
        except Exception:
            rm_file(destination)
            #print(f"relative SQL folder exists {os.path.isdir('sql')}")
            #print(f"backend SQL folder exists {os.path.isdir(backend_args.sql_path)}")
            raise HTTPException(status_code=500, detail="There was an error copying the given file into server storage.")
        finally:
            file.file.close()   
            
    def rm_file(file_path: str):
        """Removes file from file system.

        Args:
            file_path (str): _description_

        Returns:
            tuple: status_code, message
        """
        try:
            os.remove(file_path)
            status_code = 200
            message = "ok"
        except OSError as error:
            status_code = 404
            message = error
        except:
            status_code = 500
            message = f"Couldn't remove {file_path}"
            
        return status_code, message
        
    def rm_dir(folder_path: str):
        """Removes directory and contents from file system.

        Args:
            folder_path (str): _description_

        Returns:
            tuple: status_code, message
        """
        try:
            shutil.rmtree(folder_path)
            status_code = 200
            message = "ok"
        except OSError as error:
            status_code = 404
            message = error
        except:
            status_code = 500
            message = f"Couldn't remove {folder_path} and its contents"
        
        return status_code, message
    
    def create_dir(dir_path: str):
        """Creates directory in file system.

        Args:
            dir_path (str): _description_

        Returns:
            tuple: status_code, message
        """
        #create path to new db dir
        try:
            os.mkdir(dir_path)
            status_code = 200
            message = "ok"
        #if it fails, remove the previously uploaded sql file
        except FileExistsError:
            status_code = 400
            message = "Directory creation failed. Rename your uploaded file or delete the pre-existing one(s)."
        except Exception:
            status_code = 500
            message = "Directory creation failed."
        
        return status_code, message
    
    def create_sql_path(file_id):
        return os.path.join(backend_args.sql_path, file_id + ".sql")
    
    #endregion Helper Functs            

    # Initialize Picard if necessary
    with PicardLauncher() if picard_args.launch_picard else nullcontext(None):
        # Get Picard model class wrapper
        if picard_args.use_picard:
            model_cls_wrapper = lambda model_cls: with_picard(
                model_cls=model_cls, picard_args=picard_args, tokenizer=tokenizer
            )
        else:
            model_cls_wrapper = lambda model_cls: model_cls

        # Initialize model
        model = model_cls_wrapper(AutoModelForSeq2SeqLM).from_pretrained(
            backend_args.model_path,
            config=config,
            cache_dir=backend_args.cache_dir,
        )

        # Initalize generation pipeline
        pipe = Text2SQLGenerationPipeline(
            model=model,
            tokenizer=tokenizer,
            db_path=backend_args.db_path,
            prefix=data_training_args.source_prefix,
            normalize_query=data_training_args.normalize_query,
            schema_serialization_type=data_training_args.schema_serialization_type,
            schema_serialization_with_db_id=data_training_args.schema_serialization_with_db_id,
            schema_serialization_with_db_content=data_training_args.schema_serialization_with_db_content,
            device=backend_args.device,
        )

        # Initialize REST API
        app = FastAPI()
        
        #enable communication to api
        origins = ["*"]
        app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        class AskResponse(BaseModel):
            query: str
            execution_results: list
        
        def response(query: str, conn: Connection) -> AskResponse:
            try:
                return AskResponse(query=query, execution_results=conn.execute(query).fetchall())
            except OperationalError as e:
                raise HTTPException(
                    status_code=500, detail=f'while executing "{query}", the following error occurred: {e.args[0]}'
                )

        @app.get("/api/ask/{db_id}/{question}")
        def ask(db_id: str, question: str):
            """_ask the database an English question_

            Args:
                db_id (str): _database name without an extension_
                question (str): _english question_

            Raises:
                HTTPException: _404 or internal error_

            Returns:
                _list_: _list of results_
            """
            try:
                outputs = pipe(
                    inputs=Text2SQLInput(utterance=question, db_id=db_id),
                    num_return_sequences=data_training_args.num_return_sequences
                )
            except OperationalError as e:
                raise HTTPException(status_code=404, detail=e.args[0])
            try:
                conn = connect(backend_args.db_path + "/" + db_id + "/" + db_id + ".sqlite")
                return [response(query=output["generated_text"], conn=conn) for output in outputs]
            finally:
                conn.close()
        
        @app.post("/api/upload/")
        def upload(file: UploadFile = File(...)):
            """_Upload a database file into the file system._

            Args:
                file (UploadFile, optional): _A file in Sqlite3 format_. Defaults to File(...).

            Raises:
                HTTPException: _500 for unkown problem_
                HTTPException: _400 for directory/file creation failure or if uploaded file isn't a database_

            Returns:
                _json_: _message_
            """
            
            #separate file name into name + ext
            db_id, file_ext = os.path.splitext(file.filename)
            
            #path to new db dir
            db_folder_path = os.path.join(backend_args.db_path, db_id)
            
            status_code, message = create_dir(db_folder_path)
            
            if(status_code != 200):
                raise HTTPException(status_code=status_code, detail=message)
            
            #path to new db file
            new_file_path = os.path.join(db_folder_path, db_id + ".sqlite")
            
            store_file(file, new_file_path, byte_mode=True) 
            
            #if uploaded file isn't a database
            if(subprocess.call(["sqlite3", f"{new_file_path}", f".schema"]) != 0):
                #remove uploaded dir and its contents
                status_code, msg = rm_dir(db_folder_path)
                
                #log msg if rm dir fails
                if(status_code != 200):
                    print(msg)
                    
                raise HTTPException(status_code=400, detail="The uploaded file isn't a database.")
            
            return {"message": f"Successfully uploaded {file.filename} to {new_file_path}"}
        
        @app.post("/api/upload/sql")
        def uploadSql(file: UploadFile = File(...)):
            """_Uploads an sql file into the file system.
            Generates an sqlite3 formatted file from the sql file.
            Stores the sqlite3 formatted file in the file system.
            Undoes operations if any step fails._ 

            Args:
                file (UploadFile, optional): _A file containing SQL_. Defaults to File(...).

            Raises:
                HTTPException: _Create database directory error_
                HTTPException: _Conversion from SQL to database file error_

            Returns:
                _json_: _message_
            """
            
            #separate file name into name + ext
            file_id, file_ext = os.path.splitext(file.filename)
            
            #path to new sql file
            sql_file_path = create_sql_path(file_id)
            
            #store sql file in proper spot
            store_file(file, sql_file_path, byte_mode=True)
            
            #path to new db dir
            db_folder_path = os.path.join(backend_args.db_path, file_id)
            
            create_dir_code, create_dir_msg = create_dir(db_folder_path)

            #if dir creation failed
            if(create_dir_code != 200):
                #rm file
                rm_file_code, rm_file_msg = rm_file(sql_file_path)
                
                #print locally any file removal error
                if(rm_file_code != 200):
                    print(rm_file_msg)
                
                #raise error
                raise HTTPException(status_code=create_dir_code, detail=create_dir_msg)
            
            db_filename = file_id + ".sqlite"
            
            #path to new db file
            db_file_path = os.path.join(db_folder_path, db_filename)
            
            #if couldn't create database file from sql
            if(subprocess.call(["sqlite3", f"{db_file_path}", f".read {sql_file_path}"]) != 0):
                #removed saved SQL file
                rm_file_code, rm_file_msg = rm_file(sql_file_path)
                
                #print locally any file removal error
                if(rm_file_code != 200):
                    print(rm_file_msg)
                    #logger.log
                
                #remove created db folder + any contents that snuck in
                rm_dir_code, rm_dir_msg = rm_dir(db_folder_path)
                
                #print locally any file removal error
                if(rm_dir_code != 200):
                    print(rm_dir_msg)
                
                raise HTTPException(status_code=400, detail="Couldn't create database file from uploaded file. Ensure an SQL file is being uploaded.")
            
               
            return {"message": f"Successfully uploaded {file.filename} to {sql_file_path} and {db_filename} to {db_file_path}"}

        @app.delete("/api/deleteSqlDb")
        def deleteSqlDb(file_name: str):
            """_Delete both the stored sql and database file by the given file_name.
            If no file to delete in either location, throws an exception._

            Args:
                file_name (str): _File name without extension_

            Returns:
                _json_: _message_
            """
            
            correct_msg = ""
            
            file_id, file_ext = os.path.splitext(file_name)
            
            #path to new db dir
            db_folder_path = os.path.join(backend_args.db_path, file_id)
            
            #rm fb folder + contents
            rm_dir_code, rm_dir_msg = rm_dir(db_folder_path)
            
            #print locally any file removal error
            if(rm_dir_code != 200):
                print(rm_dir_msg)
            else:
                correct_msg += f"Successfully deleted {file_name} in {db_folder_path}'s contents. "
            
            #path to sql file
            sql_file_path = create_sql_path(file_id)
            
            rm_file_code, rm_file_msg = rm_file(sql_file_path)
            
            #print locally any file removal error
            if(rm_file_code != 200):
                print(rm_file_msg)
            else:
                correct_msg += f"Successfully deleted {file_name} at {sql_file_path}. "
                
            #if neither deletion operations succeeded
            if(rm_file_msg != 200 and rm_dir_code != 200):
                raise HTTPException(status_code=rm_dir_code, detail=f"{rm_file_msg} {rm_dir_msg}")
            
            return {"message": correct_msg}

        @app.get("/api/getDatabases/")
        def getDatabases():
            """_Get a list of the queryable database names_

            Raises:
                HTTPException: _error in listing databases_

            Returns:
                _list_: _database names as strings_
            """
            try:
                #get all names in db folder
                db_folders = os.listdir(backend_args.db_path)
            except:
                raise HTTPException(status_code=500, detail="There was an error when attempting to list all the database folders.")
            
            #take only the directories within the db folder names
            return [name for name in db_folders if os.path.isdir(os.path.join(backend_args.db_path, name))]
        
        @app.get("/api/getDatabases/{file_name}", responses={200: {"content": {"application/vnd.sqlite3" : {"example": "No example available."}}}})
        def getDatabaseFile(file_name: str ):
            """_Retrieve a database file by name_

            Args:
                file_name (str): _description_

            Raises:
                HTTPException: _Error 404 that requested database file couldn't be found_

            Returns:
                _FileResponse_: _database file_
            """
            
            #separate file name into name + ext (just incase sent w/ an ext)
            db_id, file_ext = os.path.splitext(file_name)
            
            db_file_name = db_id + ".sqlite"
            
            #construct path to db file
            db_file_path = os.path.join(backend_args.db_path, db_id, db_file_name)
            
            #if file exists, return it
            if(os.path.exists(db_file_path)):
                return FileResponse(db_file_path, media_type="application/vnd.sqlite3", filename=f"{db_file_name}")
            else:
                raise HTTPException(status_code=404, detail=f"Database file {file_name} not found at {db_file_path}.")

        # Run app
        run(app=app, host=backend_args.host, port=backend_args.port, forwarded_allow_ips='*')


if __name__ == "__main__":
    main()
