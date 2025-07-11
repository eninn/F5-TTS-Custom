import logging, os, time, datetime, pytz, requests
from functools import wraps

def seoul_time(timestamp):
    dt = datetime.datetime.fromtimestamp(timestamp, pytz.timezone('Asia/Seoul'))
    return dt.timetuple()
class loggerConfig:
    def __init__(self, logger_name:str, log_dir:str, log_type='file', is_print:bool=False) -> None:
        os.makedirs(log_dir, exist_ok=True)
        self.formatter = logging.Formatter('[%(asctime)s][%(levelname)s]: %(message)s')
        self.formatter.converter = seoul_time
        logging.getLogger(logger_name).handlers.clear()
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(logging.INFO)
        self.is_print = is_print
        self.logger.propagate = False

        if log_type == 'file' or log_type == 'both':
            file_handler = logging.FileHandler(os.path.join(log_dir, f'{logger_name}.log'), mode='w') # mode a 하면 추가쓰기.
            file_handler.setFormatter(self.formatter)
            self.logger.addHandler(file_handler)
        if log_type == 'stream' or log_type == 'both':
            stream_handler = logging.StreamHandler()
            stream_handler.setFormatter(self.formatter)
            self.logger.addHandler(stream_handler)

        self.logger.info(f'Logger: {logger_name} is set by {log_type}.')

    def get_formmatted_message(self, log_message):
        formatted_message = self.formatter.format(
                logging.LogRecord(
                    name=self.logger.name,
                    level=logging.INFO,
                    pathname="",
                    lineno=0,
                    msg=log_message,
                    args=(),
                    exc_info=None
                )
            )
        return formatted_message

    def add_info(self, method: str, message: str):
        self.logger.info(f'[{method}]: {message}')
        if self.is_print:
            print(self.get_formmatted_message(message), flush=True)

    def add_warning(self, method: str, message: str):
        self.logger.warning(f'[{method}]: {message}')
        if self.is_print:
            print(self.get_formmatted_message(message), flush=True)

    def add_error(self, method: str, message: str, exc_info=None):
        self.logger.error(f'[{method}]: {message}', exc_info=exc_info)
        if self.is_print:
            print(self.get_formmatted_message(message), flush=True)

    def add_exception(self, method: str, message: str):
        self.logger.exception(f'[{method}]: {message}', exc_info=True)
        if self.is_print:
            print(self.get_formmatted_message(message), flush=True)

    def add_critical(self, method: str, message: str, exc_info=None):
        self.logger.critical(f'[{method}]: {message}', exc_info=exc_info)
        if self.is_print:
            print(self.get_formmatted_message(message), flush=True)


def timeit(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f"{func.__name__} took: {end - start:.6f} seconds")
        return result
    return wrapper



def send_telegram_message(bot_token, chat_id, message):
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": message
    }
    try:
        response = requests.post(url, data=payload)
        if response.status_code == 200:
            print("메시지 전송 성공!")
        else:
            print(f"에러 발생: {response.status_code}, {response.text}")
    except Exception as e:
        print(f"요청 중 에러 발생: {e}")