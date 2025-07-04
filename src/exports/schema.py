'''
Utilities for processing export schemata.
'''
from typing import TYPE_CHECKING, Any

from pydantic import GetCoreSchemaHandler
from pydantic_core import core_schema

import logging

logger = logging.getLogger(__name__)

class Unknown:
    '''
    Always-validate type for unknown fields. Allows us to hook into it to
    print debug information about unknown fields.
    '''
    @classmethod
    def __get_pydantic_core_schema__(cls, source: Any, handler: GetCoreSchemaHandler) -> core_schema.CoreSchema:
        def unknown_validator(value: Any, info: core_schema.ValidationInfo) -> Any:
            logger.debug(f"Unknown field {info.field_name} with value: {value!r}")
            return value
        return core_schema.with_info_plain_validator_function(
            unknown_validator, field_name=handler.field_name
        )

if TYPE_CHECKING:
    type Invalid[T] = T
    '''
    Always-invalid type for invalid fields. Allows us to switch the fallback
    type to this to ensure we can catch unaccounted-for cases.
    '''
else:
    class Invalid[T]:
        '''
        Always-invalid type for invalid fields. Allows us to switch the fallback
        type to this to ensure we can catch unaccounted-for cases.
        '''
        @classmethod
        def __get_pydantic_core_schema__(cls, source, handler: GetCoreSchemaHandler) -> core_schema.CoreSchema:
            def invalid_validator(value, info: core_schema.ValidationInfo):
                raise ValueError(
                    f"Invalid field {info.field_name} with value: {value!r}"
                )
            if False and logger.getEffectiveLevel() > logging.DEBUG:
                return handler(source.__args__[0])
            return core_schema.with_info_plain_validator_function(
                invalid_validator, field_name=handler.field_name
            )