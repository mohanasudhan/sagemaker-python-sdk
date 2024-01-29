
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.
"""Holds mixin logic to support deployment of Model ID"""
from __future__ import absolute_import
import logging
import os
from pathlib import Path
from typing import Type
from abc import ABC, abstractmethod
from sagemaker.serve.detector.pickler import save_pkl
from sagemaker.base_predictor import PredictorBase
from sagemaker.serve.model_server.fastapi.prepare import prepare_for_fastapi
from sagemaker.serve.mode.function_pointers import Mode
from sagemaker.serve.mode.sagemaker_endpoint_mode import SageMakerEndpointMode
from sagemaker.serve.mode.local_container_mode import LocalContainerMode
from sagemaker.serve.utils.telemetry_logger import _capture_telemetry
from sagemaker.serve.utils.predictors import _get_local_mode_predictor
from sagemaker.serve.validations.check_image_uri import is_1p_image_uri
from sagemaker.serve.validations.check_image_and_hardware_type import (
    validate_image_uri_and_hardware,
)
from sagemaker.model import Model
import cloudpickle

logger = logging.getLogger(__name__)

class FastAPIServe(ABC):

    def __init__(self):
        self.model = None
        self.serve_settings = None
        self.sagemaker_session = None
        self.model_path = None
        self.dependencies = None
        self.modes = None
        self.mode = None
        self.model_server = None
        self.image_uri = None
        self._original_deploy = None
        self.hf_model_config = None
        self._default_tensor_parallel_degree = None
        self._default_data_type = None
        self._default_max_tokens = None
        self.pysdk_model = None
        self.schema_builder = None
        self.env_vars = None
        self.nb_instance_type = None
        self.ram_usage_model_load = None
        self.secret_key = None
        self.jumpstart = None
        self.role_arn = None

    @abstractmethod
    def _prepare_for_mode(self):
        """Placeholder docstring"""

    @abstractmethod
    def _get_client_translators(self):
        """Placeholder docstring"""

    def _fast_api_save_model_inference_spec(self):
        """Placeholder docstring"""
        # check if path exists and create if not
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

        code_path = Path(self.model_path)
        # save the model or inference spec in cloud pickle format
        if self.inference_spec:
            save_pkl(code_path, (self.inference_spec, self.schema_builder))
            self.image_uri = "027412998179.dkr.ecr.us-west-2.amazonaws.com/langchain-serve-container:latest"
        elif self.model:
            self.env_vars.update(
                {
                    "MODEL_CLASS_NAME": f"{self.model.__class__.__module__}.{self.model.__class__.__name__}"
                }
            )
            self.image_uri = "027412998179.dkr.ecr.us-west-2.amazonaws.com/langchain-serve-container:latest"
            cloudpickle.register(ssl.SSLContext, lambda _: None)
            save_pkl(code_path, (self.model, self.schema_builder))
        else:
            raise ValueError("Cannot detect required model or inference spec")
        
    
    def _fast_api_prepare_for_mode(self):
        """Placeholder docstring"""
        # TODO: move mode specific prepare steps under _model_builder_deploy_wrapper
        self.s3_upload_path = None
        if self.mode == Mode.SAGEMAKER_ENDPOINT:
            # init the SageMakerEndpointMode object
            self.modes[str(Mode.SAGEMAKER_ENDPOINT)] = SageMakerEndpointMode(
                inference_spec=self.inference_spec, model_server=self.model_server
            )
            self.s3_upload_path, env_vars_sagemaker = self.modes[
                str(Mode.SAGEMAKER_ENDPOINT)
            ].prepare(
                self.model_path,
                self.secret_key,
                self.serve_settings.s3_model_data_url,
                self.sagemaker_session,
                self.image_uri,
                self.jumpstart if hasattr(self, "jumpstart") else False,
            )
            self.env_vars.update(env_vars_sagemaker)
            return self.s3_upload_path, env_vars_sagemaker
        if self.mode == Mode.LOCAL_CONTAINER:
            # init the LocalContainerMode object
            self.modes[str(Mode.LOCAL_CONTAINER)] = LocalContainerMode(
                inference_spec=self.inference_spec,
                schema_builder=self.schema_builder,
                session=self.sagemaker_session,
                model_path=self.model_path,
                env_vars=self.env_vars,
                model_server=self.model_server,
            )
            self.modes[str(Mode.LOCAL_CONTAINER)].prepare()
            return None

        raise ValueError(
            "Please specify mode in: %s, %s" % (Mode.LOCAL_CONTAINER, Mode.SAGEMAKER_ENDPOINT)
        )
    
    def _fast_api_create_model(self):
        """Placeholder docstring"""
        # TODO: we should create model as per the framework
        self.pysdk_model = Model(
            image_uri=self.image_uri,
            model_data=self.s3_upload_path,
            role=self.serve_settings.role_arn,
            env=self.env_vars,
            sagemaker_session=self.sagemaker_session,
            predictor_cls=self._get_predictor,
        )

        # store the modes in the model so that we may
        # reference the configurations for local deploy() & predict()
        self.pysdk_model.mode = self.mode
        self.pysdk_model.modes = self.modes
        self.pysdk_model.serve_settings = self.serve_settings

        # dynamically generate a method to direct model.deploy() logic based on mode
        # unique method to models created via ModelBuilder()
        self._original_deploy = self.pysdk_model.deploy
        self.pysdk_model.deploy = self._fast_api_model_builder_deploy_wrapper
        self._original_register = self.pysdk_model.register
        self.pysdk_model.register = self._fast_api_model_builder_register_wrapper
        self.model_package = None
        return self.pysdk_model

    @_capture_telemetry("register")
    def _fast_api_model_builder_register_wrapper(self, *args, **kwargs):
        """Placeholder docstring"""
        serializer, deserializer = self._get_client_translators()
        if "content_types" not in kwargs:
            self.pysdk_model.content_types = serializer.CONTENT_TYPE.split()
        if "response_types" not in kwargs:
            self.pysdk_model.response_types = deserializer.ACCEPT.split()
        new_model_package = self._original_register(*args, **kwargs)
        self.pysdk_model.model_package_arn = new_model_package.model_package_arn
        new_model_package.deploy = self._fast_api_model_builder_deploy_model_package_wrapper
        self.model_package = new_model_package
        return new_model_package

    def _fast_api_model_builder_deploy_model_package_wrapper(self, *args, **kwargs):
        """Placeholder docstring"""
        if self.pysdk_model.model_package_arn is not None:
            return self._fast_api_model_builder_deploy_wrapper(*args, **kwargs)

        # need to set the model_package_arn
        # so that the model is created using the model_package's configs
        self.pysdk_model.model_package_arn = self.model_package.model_package_arn
        predictor = self._fast_api_model_builder_deploy_wrapper(*args, **kwargs)
        self.pysdk_model.model_package_arn = None
        return predictor

    @_capture_telemetry("fastapi.deploy")
    def _fast_api_model_builder_deploy_wrapper(
        self,
        *args,
        container_timeout_in_second: int = 300,
        instance_type: str = None,
        initial_instance_count: int = None,
        mode: str = None,
        **kwargs,
    ) -> Type[PredictorBase]:
        """Placeholder docstring"""
        if mode and mode != self.mode:
            self._fast_api_overwrite_mode_in_deploy(overwrite_mode=mode)

        if self.mode == Mode.LOCAL_CONTAINER:
            serializer, deserializer = self._get_client_translators()
            predictor = _get_local_mode_predictor(
                mode_obj=self.modes[str(Mode.LOCAL_CONTAINER)],
                model_server=self.model_server,
                serializer=serializer,
                deserializer=deserializer,
            )

            self.modes[str(Mode.LOCAL_CONTAINER)].create_server(
                self.image_uri, container_timeout_in_second, self.secret_key, predictor
            )
            return predictor
        if self.mode == Mode.SAGEMAKER_ENDPOINT:
            # Validate parameters
            if not instance_type:
                raise ValueError("Missing required parameter `instance_type`")

            if not initial_instance_count:
                raise ValueError("Missing required parameter `initial_instance_count`")

            if is_1p_image_uri(image_uri=self.image_uri):
                validate_image_uri_and_hardware(
                    image_uri=self.image_uri,
                    instance_type=instance_type,
                    model_server=self.model_server,
                )

        if "endpoint_logging" not in kwargs:
            kwargs["endpoint_logging"] = True
        return self._original_deploy(
            *args,
            instance_type=instance_type,
            initial_instance_count=initial_instance_count,
            **kwargs,
        )

    def _fast_api_overwrite_mode_in_deploy(self, overwrite_mode: str):
        """Mode overwritten by customer during model.deploy()"""
        logger.warning(
            "Deploying in %s Mode, overriding existing configurations set for %s mode",
            overwrite_mode,
            self.mode,
        )
        if overwrite_mode == Mode.SAGEMAKER_ENDPOINT:
            self.mode = self.pysdk_model.mode = Mode.SAGEMAKER_ENDPOINT
            s3_upload_path, env_vars_sagemaker = self._fast_api_prepare_for_mode()
            self.pysdk_model.model_data = s3_upload_path
            self.pysdk_model.env.update(env_vars_sagemaker)

        elif overwrite_mode == Mode.LOCAL_CONTAINER:
            self.mode = self.pysdk_model.mode = Mode.LOCAL_CONTAINER
            self._fast_api_prepare_for_mode()
        else:
            raise ValueError("Mode %s is not supported!" % overwrite_mode)

    """FastAPIServe build logic for ModelBuilder()"""
    def _build_for_fastapi(self) -> Type[Model]:
        """Build the model for fastapi"""
        self._fast_api_save_model_inference_spec()

        self.secret_key = prepare_for_fastapi(
            model_path=self.model_path,
            shared_libs=self.shared_libs,
            dependencies=self.dependencies,
            session=self.sagemaker_session,
            image_uri=self.image_uri,
            inference_spec=self.inference_spec,
        )

        self._fast_api_prepare_for_mode()

        return self._fast_api_create_model()
