# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

if((USE_SOMA_CODEGEN STREQUAL "ON") OR (USE_SOMA_CODEGEN STREQUAL "JSON"))
  # This branch is currently not used, enable this if we want to use the
  # JSON codegen.
  add_definitions(-DUSE_JSON_RUNTIME=1)
  file(GLOB SOMA_RELAY_CONTRIB_SRC src/relay/backend/contrib/soma/*.cc)
  list(APPEND COMPILER_SRCS ${SOMA_RELAY_CONTRIB_SRC})
  # list(APPEND COMPILER_SRCS ${JSON_RELAY_CONTRIB_SRC})

  find_library(EXTERN_LIBRARY_SOMA soma)
  list(APPEND TVM_RUNTIME_LINKER_LIBS ${EXTERN_LIBRARY_SOMA})
  # file(GLOB SOMA_CONTRIB_SRC src/runtime/contrib/soma/soma_json_runtime.cc)
  list(APPEND RUNTIME_SRCS ${SOMA_CONTRIB_SRC})
  message(STATUS "Build with SOMA JSON runtime: " ${EXTERN_LIBRARY_SOMA})
elseif(USE_SOMA_CODEGEN STREQUAL "C_SRC")
  file(GLOB SOMA_RELAY_CONTRIB_SRC src/relay/backend/contrib/soma/*.cc)
  list(APPEND COMPILER_SRCS ${SOMA_RELAY_CONTRIB_SRC})

  # TODO: We can package the soma library together with this SOMA TVM branch.
  # find_library(EXTERN_LIBRARY_SOMA soma)
  #list(APPEND TVM_RUNTIME_LINKER_LIBS ${EXTERN_LIBRARY_SOMA})
  file(GLOB SOMA_CONTRIB_SRC src/runtime/contrib/soma/soma.cc)

  #list(APPEND RUNTIME_SRCS ${SOMA_CONTRIB_SRC})
  #message(STATUS "Build with SOMA C source module: " ${EXTERN_LIBRARY_SOMA})
endif()

