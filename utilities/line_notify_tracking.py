import requests
from datetime import datetime
import numpy as np


class Send_notify_to_line:
    def __init__(
        self,
        line_token: str,
        exp_name: str,
        model: str,
        batch_size: int,
        loss: str,
        optimizer: str,
        lr: float,
        total_epoch: int,
    ):
        self.token = line_token
        self.exp_name = exp_name
        self.model = model
        self.batch_size = batch_size
        self.loss = loss
        self.optimizer = optimizer
        self.lr = lr
        self.total_epoch = total_epoch
        self.url = "https://notify-api.line.me/api/notify"

    def send_message(self, current_epoch):
        epoch_range = np.arange(0, self.total_epoch)

        p_0 = int(np.percentile(epoch_range, 0))
        p_25 = int(np.percentile(epoch_range, 25))
        p_50 = int(np.percentile(epoch_range, 50))
        p_75 = int(np.percentile(epoch_range, 75))
        p_100 = int(np.percentile(epoch_range, 100))

        if current_epoch in [p_0, p_25, p_50, p_75, p_100]:
            message = f"""
Timestamp: {datetime.now().strftime("%Y-%m-%d %H:%M")}
Exp_name: {self.exp_name}
Model: {self.model}
Batch_size: {self.batch_size}
Loss: {self.loss}
Optimizer: {self.optimizer}
Learning Rate: {self.lr}
Total Epochs: {self.total_epoch}
Training progess: {round((current_epoch+1)*100 / (self.total_epoch))}% 
({current_epoch} from {self.total_epoch-1})
        """

            data = {"message": message}
            headers = {"Authorization": "Bearer " + self.token}
            session = requests.Session()
            session_post = session.post(self.url, headers=headers, data=data)

    def send_error(self, error_message):
        error_message = f"""
An error occured in
Exp_name: {self.exp_name}
Error: {error_message}
"""
        data = {"message": error_message}
        headers = {"Authorization": "Bearer " + self.token}
        session = requests.Session()
        session_post = session.post(self.url, headers=headers, data=data)

