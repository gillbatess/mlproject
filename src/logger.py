import logging
import os
from datetime import datetime

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

LOG_FILE = datetime.now().strftime("%Y_%m_%d_%H_%M_%S.log")

logging.basicConfig(
    filename=os.path.join(LOG_DIR, LOG_FILE),
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(filename)s:%(lineno)d | %(message)s",
)

