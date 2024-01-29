from __future__ import absolute_import
import os
import cloudpickle
import shutil
import platform
import logging
from pathlib import Path
from functools import partial
from fastapi import FastAPI, Request
from sagemaker.serve.validations.check_integrity import perform_integrity_check
from sagemaker.serve.spec.inference_spec import InferenceSpec
from sagemaker.serve.detector.image_detector import _detect_framework_and_version, _get_model_base
from sagemaker.serve.detector.pickler import load_xgboost_from_json


logger = logging.getLogger(__name__)

inference_spec = None
native_model = None
schema_builder = None
model_dir = "/opt/ml/model"
app = FastAPI()

@app.get('/ping')
async def ping():
    healthy = False
    try:
        if predict_callable is not None:
            healthy = True
    except:
        logger.error("Model not loaded yet or issue with loading {}".format(predict_callable))

    return {"status": "Healthy"} if healthy else {"status": "Unhealthy"}


@app.on_event('startup')
def load():
    _run_preflight_diagnostics()
    _install_requirements()
    shared_libs_path = Path(model_dir + "/shared_libs")

    if shared_libs_path.exists():
        # before importing, place dynamic linked libraries in shared lib path
        shutil.copytree(shared_libs_path, "/lib", dirs_exist_ok=True)

    serve_path = Path(__file__).parent.joinpath("serve.pkl")
    with open(str(serve_path), mode="rb") as file:
        global inference_spec, native_model, schema_builder, predict_callable
        obj = cloudpickle.load(file)
        if isinstance(obj[0], InferenceSpec):
            inference_spec, schema_builder = obj
        else:
            native_model, schema_builder = obj

            from langchain.load.dump import dumps, dumpd
            native_model = dumpd(native_model)

    if native_model:
        framework, _ = _detect_framework_and_version(
            model_base=str(_get_model_base(model=native_model))
        )
        if framework == "pytorch":
            native_model.eval()
        predict_callable = native_model if callable(native_model) else native_model.predict
    elif inference_spec:
        predict_callable = partial(inference_spec.invoke, model=inference_spec.load(model_dir))
    
    logger.info("Model loaded successfully")

    return {"status": "Loaded"}


@app.post('/invocations')
async def invocations(request: Request):
    request_body = await request.body()

    logger.info(f"Invoked with payload {request_body} of size: {len(request_body)}")
 
    return predict_callable(request_body)

@app.on_event('shutdown')
def shutdown():
    # emit some metrics
    return {"status": "ShutDown"}


def _run_preflight_diagnostics():
    _py_vs_parity_check()
    _pickle_file_integrity_check()


def _py_vs_parity_check():
    container_py_vs = platform.python_version()
    local_py_vs = os.getenv("LOCAL_PYTHON")

    if not local_py_vs or container_py_vs.split(".")[1] != local_py_vs.split(".")[1]:
        logger.warning(
            f"The local python version {local_py_vs} differs from the python version "
            f"{container_py_vs} on the container. Please align the two to avoid unexpected behavior"
        )


def _pickle_file_integrity_check():
    with open("/opt/ml/model/serve.pkl", "rb") as f:
        buffer = f.read()

    metadeata_path = Path("/opt/ml/model/metadata.json")
    perform_integrity_check(buffer=buffer, metadata_path=metadeata_path)


def _install_requirements():
    requirements_path = Path("/opt/ml/model/requirements.txt")
    if requirements_path.exists():
        os.system("pip install -r {}".format(requirements_path))