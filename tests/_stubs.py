import sys
import types


def install_requests_stub():
    if "requests" in sys.modules:
        return

    try:
        import requests  # noqa: F401
        return
    except ModuleNotFoundError:
        pass

    requests_mod = types.ModuleType("requests")
    auth_mod = types.ModuleType("requests.auth")

    class Session:
        pass

    class Response:
        pass

    class HTTPDigestAuth:
        def __init__(self, username, password):
            self.username = username
            self.password = password

    requests_mod.Session = Session
    requests_mod.Response = Response
    auth_mod.HTTPDigestAuth = HTTPDigestAuth
    requests_mod.auth = auth_mod

    sys.modules["requests"] = requests_mod
    sys.modules["requests.auth"] = auth_mod


def install_pipeline_import_stubs():
    install_requests_stub()

    if "cv2" not in sys.modules:
        cv2_mod = types.ModuleType("cv2")
        cv2_mod.CAP_PROP_BUFFERSIZE = 38
        cv2_mod.CAP_PROP_FPS = 5

        def imwrite(path, _image):
            with open(path, "wb") as handle:
                handle.write(b"jpg")
            return True

        cv2_mod.imwrite = imwrite
        sys.modules["cv2"] = cv2_mod

    if "numpy" not in sys.modules:
        numpy_mod = types.ModuleType("numpy")

        class ndarray:
            pass

        numpy_mod.ndarray = ndarray
        numpy_mod.uint8 = "uint8"
        sys.modules["numpy"] = numpy_mod

    if "yaml" not in sys.modules:
        yaml_mod = types.ModuleType("yaml")
        yaml_mod.safe_load = lambda *_args, **_kwargs: {}
        sys.modules["yaml"] = yaml_mod

    if "telegram" not in sys.modules:
        telegram_mod = types.ModuleType("telegram")

        class Update:
            pass

        class InlineKeyboardButton:
            def __init__(self, *args, **kwargs):
                self.args = args
                self.kwargs = kwargs

        class InlineKeyboardMarkup:
            def __init__(self, keyboard):
                self.keyboard = keyboard

        telegram_mod.Update = Update
        telegram_mod.InlineKeyboardButton = InlineKeyboardButton
        telegram_mod.InlineKeyboardMarkup = InlineKeyboardMarkup
        sys.modules["telegram"] = telegram_mod

    if "telegram.ext" not in sys.modules:
        telegram_ext_mod = types.ModuleType("telegram.ext")

        class ContextTypes:
            DEFAULT_TYPE = object

        telegram_ext_mod.ContextTypes = ContextTypes
        sys.modules["telegram.ext"] = telegram_ext_mod
