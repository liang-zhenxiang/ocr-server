#  Copyright(c) 2020-2024, www.flydiy.cn Ltd. All rights reserved.

import logging
import os
import sys
from http import HTTPStatus
from pathlib import Path
from typing import Optional, Any, TYPE_CHECKING

from fastapi import File, Form, UploadFile
from pydantic import BaseModel, Field, ConfigDict

if TYPE_CHECKING:
    try:
        from rapidocr_paddle import RapidOCR
    except ImportError:
        from rapidocr_onnxruntime import RapidOCR


_logger = logging.getLogger(__name__)


def _log(msg: str):
    _logger.debug(msg)
    # print(f"[{datetime.now()}] [{os.getpid()}] {msg}")


sys.path.append(os.path.dirname(os.path.dirname(__file__)))

_log("Start Import Controller")

from starlette.responses import RedirectResponse
from fastapi import FastAPI
from fastapi.responses import JSONResponse

_log("End Import Controller")


class BaseResponse(BaseModel):
    code: int = Field(200, description="API status code")
    msg: str = Field("success", description="API status message")
    data: Any = Field(None, description="API data")

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "code": 200,
            "msg": "success",
        }
    })


def get_root_path() -> str:
    """
    获取项目根路径
    :return:
    """
    # 获取文件目录
    cur_path = os.path.abspath(os.path.dirname(__file__))

    # 获取项目根路径，内容为当前项目的名字
    root_path = os.path.abspath(os.path.join(cur_path, "../"))
    return root_path


def make_fastapi_offline(
        fast_app: FastAPI,
        static_dir=Path(get_root_path()).joinpath("static"),
        static_url="/static-offline-docs",
        docs_url: Optional[str] = "/docs",
        redoc_url: Optional[str] = "/redoc",
) -> None:
    """patch the FastAPI obj that doesn't rely on CDN for the documentation page"""
    from fastapi import Request
    from fastapi.openapi.docs import (
        get_redoc_html,
        get_swagger_ui_html,
        get_swagger_ui_oauth2_redirect_html,
    )
    from fastapi.staticfiles import StaticFiles

    openapi_url = fast_app.openapi_url
    swagger_ui_oauth2_redirect_url = fast_app.swagger_ui_oauth2_redirect_url

    def remove_route(url: str) -> None:
        """
        remove original route from app
        """
        index = None
        for i, r in enumerate(fast_app.routes):
            if r.path.lower() == url.lower():
                index = i
                break
        if isinstance(index, int):
            fast_app.routes.pop(index)

    # Set up static file mount
    fast_app.mount(
        static_url,
        StaticFiles(directory=Path(static_dir).as_posix()),
        name="static-offline-docs",
    )

    if docs_url is not None:
        from starlette.responses import HTMLResponse

        remove_route(docs_url)
        remove_route(swagger_ui_oauth2_redirect_url)

        # Define the doc and redoc pages, pointing at the right files
        @fast_app.get(docs_url, include_in_schema=False)
        async def custom_swagger_ui_html(request: Request) -> HTMLResponse:
            root = request.scope.get("root_path")
            favicon = f"{root}{static_url}/favicon.png"
            return get_swagger_ui_html(
                openapi_url=f"{root}{openapi_url}",
                title=fast_app.title + " - Swagger UI",
                oauth2_redirect_url=swagger_ui_oauth2_redirect_url,
                swagger_js_url=f"{root}{static_url}/swagger-ui-bundle.js",
                swagger_css_url=f"{root}{static_url}/swagger-ui.css",
                swagger_favicon_url=favicon,
            )

        @fast_app.get(swagger_ui_oauth2_redirect_url, include_in_schema=False)
        async def swagger_ui_redirect() -> HTMLResponse:
            return get_swagger_ui_oauth2_redirect_html()

    if redoc_url is not None:
        from starlette.responses import HTMLResponse
        remove_route(redoc_url)

        @fast_app.get(redoc_url, include_in_schema=False)
        async def redoc_html(request: Request) -> HTMLResponse:
            root = request.scope.get("root_path")
            favicon = f"{root}{static_url}/favicon.png"

            return get_redoc_html(
                openapi_url=f"{root}{openapi_url}",
                title=fast_app.title + " - ReDoc",
                redoc_js_url=f"{root}{static_url}/redoc.standalone.js",
                with_google_fonts=False,
                redoc_favicon_url=favicon,
            )


def mount_app_routes(fast_app: FastAPI):

    @fast_app.get(path="/", summary="swagger 文档")
    async def index():
        return RedirectResponse(url="/docs")

    @fast_app.get(path="/actuator/health", summary="健康检测端点")
    async def actuator_health():
        return JSONResponse(
            status_code=HTTPStatus.OK.value,
            content={'status': 'UP'}
        )


def create_app(fast_app: FastAPI = None):
    _log("START mount_routes")

    if fast_app is None:
        fast_app = FastAPI(
            title="FlyGPT-Engine Controller Extend API Server",
            version="v0.1.0"
        )

    make_fastapi_offline(fast_app)

    # 挂载路由

    _log("mount_app_routes")
    mount_app_routes(fast_app)

    _log("mount_ocr_routes")
    fast_app.post("/extend/ocr", tags=["Chat"], summary="图转文")(ocr)

    _log("END mount_routes")

    return fast_app


def ocr(file: UploadFile = File(..., description="上传文件"),
        type: str = Form(..., description="文件类型", examples=["pdf、image"])) -> BaseResponse:
    try:
        file_content = file.file.read()
        result = get_file_content_by_ocr(type, file_content)
        return BaseResponse(code=200, msg="已经图转文", data=result)
    except Exception as e:
        err_msg = f"========调用OCR文件识别功能失败  \r\n错误详情：{str(e)}"
        _logger.error(err_msg)
        return BaseResponse(code=500, msg=err_msg)


def _get_ocr(use_cuda: bool = True) -> "RapidOCR":
    try:
        from rapidocr_paddle import RapidOCR
        ocr_object = RapidOCR(det_use_cuda=use_cuda, cls_use_cuda=use_cuda, rec_use_cuda=use_cuda)
    except (ImportError, ValueError) as e:
        _logger.error(f"===rapidocr_paddle GPU版本出错：{e}")
        from rapidocr_onnxruntime import RapidOCR
        ocr_object = RapidOCR()
    return ocr_object


def get_file_content_by_ocr(file_type, file_bytes: bytes):
    """
        通过OCR识别文件内容
        Args:
            file_type: 文件类型，包括：pdf、image
            file_bytes: 文件二进制

        Returns:
            string：文件内容
        """

    if file_type == "image":
        return _img2text(file_bytes)
    elif file_type == "pdf":
        return _pdf2text(file_bytes)


def _pdf2text(file_bytes: bytes) -> str:
    import fitz  # pyMuPDF里面的fitz包，不要与pip install fitz混淆
    import tqdm
    import numpy as np
    ocr_object = _get_ocr()
    doc = fitz.open(stream=file_bytes)
    resp = ""

    b_unit = tqdm.tqdm(total=doc.page_count, desc="调用飞桨RapidOCR解析进度: 0")
    for i, page in enumerate(doc):

        # 更新描述
        b_unit.set_description("调用飞桨RapidOCR解析进度: {}".format(i))
        # 立即显示进度条更新结果
        b_unit.refresh()
        #  TODO: 依据文本与图片顺序调整处理方式
        text = page.get_text("")
        resp += text + "\n"

        img_list = page.get_images()
        for img in img_list:
            pix = fitz.Pixmap(doc, img[0])
            img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, -1)
            result, _ = ocr_object(img_array)
            if result:
                ocr_result = [line[1] for line in result]
                resp += "\n".join(ocr_result)

        # 更新进度
        b_unit.update(1)
    return resp


def _img2text(file_bytes: bytes) -> str:
    resp = ""
    ocr_object = _get_ocr()
    result, _ = ocr_object(file_bytes)
    if result:
        ocr_result = [line[1] for line in result]
        resp += "\n".join(ocr_result)
    return resp


def run_api(host, port, **kwargs):
    import uvicorn
    if kwargs.get("ssl_keyfile") and kwargs.get("ssl_certfile"):
        uvicorn.run(app,
                    host=host,
                    port=port,
                    ssl_keyfile=kwargs.get("ssl_keyfile"),
                    ssl_certfile=kwargs.get("ssl_certfile"),
                    )
    else:
        uvicorn.run(app, host=host, port=port)


if __name__ == '__main__':
    import uvicorn
    import argparse

    parser = argparse.ArgumentParser(prog='langchain-OCR',
                                     description='About langchain-OCR'
                                                 ' ｜ 基于OCR的文本识别')
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=20001)
    parser.add_argument("--ssl_keyfile", type=str)
    parser.add_argument("--ssl_certfile", type=str)
    # 初始化消息
    args = parser.parse_args()
    args_dict = vars(args)

    # 初始化消息
    app = create_app()

    run_api(
        host=args.host,
        port=args.port,
        ssl_keyfile=args.ssl_keyfile,
        ssl_certfile=args.ssl_certfile,
    )
