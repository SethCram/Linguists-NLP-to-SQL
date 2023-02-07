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
from uvicorn import run
from sqlite3 import Connection, connect, OperationalError
from seq2seq.utils.pipeline import Text2SQLGenerationPipeline, Text2SQLInput, get_schema
from seq2seq.utils.picard_model_wrapper import PicardArguments, PicardLauncher, with_picard
from seq2seq.utils.dataset import DataTrainingArguments
import shutil

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

        @app.get("/ask/{db_id}/{question}")
        def ask(db_id: str, question: str):
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
        
        @app.post("/upload/")
        def upload(file: UploadFile = File(...)):
            
            #separate file name into name + ext
            db_id, file_ext = os.path.splitext(file.filename)
            
            #path to new db dir
            db_folder_path = os.path.join(backend_args.db_path, db_id)
            
            try:
                os.mkdir(db_folder_path)
            except FileExistsError:
                raise HTTPException(status_code=400, detail="Directory creation failed. Database using that same name has probably already been uploaded. Rename your database file.")
            except Exception:
                raise HTTPException(status_code=500, detail="Directory creation failed.")
            
            #path to new db file
            new_file_path = os.path.join(db_folder_path, db_id + ".sqlite")
            
            try:
                with open(new_file_path, 'wb') as file_obj:
                    shutil.copyfileobj(file.file, file_obj)
            except Exception:
                raise HTTPException(status_code=500, detail="There was an error copying the given file into server storage.")
            finally:
                file.file.close()    
            return {"message": f"Successfully uploaded {file.filename} to {new_file_path}"}

        @app.get("/getDatabases/")
        def getDatabases():
            try:
                #get all names in db folder
                db_folders = os.listdir(backend_args.db_path)
            except:
                raise HTTPException(status_code=500, detail="There was an error when attempting to list all the database folders.")
            
            #take only the directories within the db folder names
            return [name for name in db_folders if os.path.isdir(os.path.join(backend_args.db_path, name))]
        
        @app.get("/getDatabases/{file_name}")
        def getDatabaseFile(file_name: str, responses={200: {"content": {"application/vnd.sqlite3" : {"example": "No example available."}}}}):
            
            #separate file name into name + ext (just incase sent w/ an ext)
            db_id, file_ext = os.path.splitext(file_name)
            
            db_file_name = db_id + ".sqlite"
            
            #construct path to db file
            db_file_path = os.path.join(backend_args.db_path, db_file_name)
            
            #if file exists, return it
            if(os.path.exists(db_file_path)):
                return FileResponse(db_file_path, mediaType="application/vnd.sqlite3", filename=f"{db_file_name}")
            else:
                raise HTTPException(status_code=404, detail="Database file not found.")

        # Run app
        run(app=app, host=backend_args.host, port=backend_args.port)


if __name__ == "__main__":
    main()
