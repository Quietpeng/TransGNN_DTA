import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import time
import json
import os

class EarlyStopping:
    """
    早停类，包含邮件操作和邮箱配置
    """

    def __init__(self, patience=10, config_path="config.json"):
        """
        初始化早停类
        :param patience: 早停的耐心值
        :param config_path: 配置文件路径
        """
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
        # 从配置文件加载邮箱设置
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            email_config = config.get("email", {})
            self.send_email_enabled = email_config.get("enabled", False)
            self.sender_email = email_config.get("sender_email", "")
            self.sender_password = email_config.get("sender_password", "")
            self.receiver_email = email_config.get("receiver_email", "")
            self.smtp_server = email_config.get("smtp_server", "")
            self.smtp_port = email_config.get("smtp_port", 465)
        except Exception as e:
            print(f"加载邮箱配置失败: {str(e)}")
            self.send_email_enabled = False
            self.sender_email = ""
            self.sender_password = ""
            self.receiver_email = ""
            self.smtp_server = ""
            self.smtp_port = 465

    def __call__(self, val_loss, mse, ci, epoch, batch_id):
        """
        每次验证时调用该方法
        :param val_loss: 验证损失
        :param mse: 当前MSE指标
        :param ci: 当前CI指标
        :param epoch: 当前训练轮数
        :param batch_id: 当前批次编号
        """
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                if self.send_email_enabled and all([self.sender_email, self.sender_password, self.receiver_email, self.smtp_server, self.smtp_port]):
                    self.send_email(val_loss, mse, ci, epoch, batch_id)
        else:
            self.best_score = score
            self.counter = 0

        return self.early_stop

    def send_email(self, val_loss, mse, ci, epoch, batch_id):
        """
        发送邮件
        :param val_loss: 验证损失
        :param mse: 当前MSE指标
        :param ci: 当前CI指标
        :param epoch: 当前训练轮数
        :param batch_id: 当前批次编号
        """
        msg = MIMEMultipart()
        msg['From'] = self.sender_email
        msg['To'] = self.receiver_email
        msg['Subject'] = "Early Stopping Notification"

        current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        body = f"Early stopping triggered!\n\n" \
               f"Validation Loss: {val_loss}\n" \
               f"MSE: {mse}\n" \
               f"CI: {ci}\n" \
               f"Epoch: {epoch}\n" \
               f"Batch ID: {batch_id}\n" \
               f"Time: {current_time}"

        msg.attach(MIMEText(body, 'plain'))

        try:
            server = smtplib.SMTP_SSL(self.smtp_server, self.smtp_port)
            server.login(self.sender_email, self.sender_password)
            text = msg.as_string()
            server.sendmail(self.sender_email, self.receiver_email, text)
            server.quit()
            print("Email sent successfully!")
        except Exception as e:
            print(f"Error sending email: {e}")

    def send_training_failure_email(self, error_message):
        """
        当训练失败时发送邮件通知
        :param error_message: 错误信息
        """
        if not self.send_email_enabled:
            print("邮件发送已禁用，不发送训练失败通知")
            return
            
        msg = MIMEMultipart()
        msg['From'] = self.sender_email
        msg['To'] = self.receiver_email
        msg['Subject'] = "Training Failure Notification"

        current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        body = f"Training failed!\n\n" \
               f"Error Message: {error_message}\n" \
               f"Time: {current_time}"

        msg.attach(MIMEText(body, 'plain'))

        try:
            server = smtplib.SMTP_SSL(self.smtp_server, self.smtp_port)
            server.login(self.sender_email, self.sender_password)
            text = msg.as_string()
            server.sendmail(self.sender_email, self.receiver_email, text)
            server.quit()
            print("Training failure email sent successfully!")
        except Exception as e:
            print(f"Error sending training failure email: {e}")

class EmailTest:
    def __init__(self, config_path="config.json"):
        # 从配置文件加载邮箱设置
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            email_config = config.get("email", {})
            self.send_email_enabled = email_config.get("enabled", False)
            self.sender_email = email_config.get("sender_email", "")
            self.sender_password = email_config.get("sender_password", "")
            self.receiver_email = email_config.get("receiver_email", "")
            self.smtp_server = email_config.get("smtp_server", "")
            self.smtp_port = email_config.get("smtp_port", 465)
        except Exception as e:
            print(f"加载邮箱配置失败: {str(e)}")
            self.send_email_enabled = False
            self.sender_email = ""
            self.sender_password = ""
            self.receiver_email = ""
            self.smtp_server = ""
            self.smtp_port = 465

    def test_email(self):
        if not self.send_email_enabled:
            print("邮件发送已禁用，不执行测试")
            return
            
        msg = MIMEMultipart()
        msg['From'] = self.sender_email
        msg['To'] = self.receiver_email
        msg['Subject'] = "Email Test Notification"

        body = "This is a test email to check the email sending functionality."
        msg.attach(MIMEText(body, 'plain'))

        try:
            server = smtplib.SMTP_SSL(self.smtp_server, self.smtp_port)
            server.login(self.sender_email, self.sender_password)
            text = msg.as_string()
            server.sendmail(self.sender_email, self.receiver_email, text)
            server.quit()
            print("Test email sent successfully!")
        except Exception as e:
            print(f"Error sending test email: {e}")

if __name__ == "__main__":
    email_test = EmailTest()
    email_test.test_email()

    # 模拟训练失败
    early_stopping = EarlyStopping()
    try:
        # 模拟训练代码
        raise ValueError("Training failed due to invalid input.")
    except Exception as e:
        early_stopping.send_training_failure_email(str(e))