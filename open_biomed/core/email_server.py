import smtplib
import os
import logging
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
from email.header import Header
from dotenv import load_dotenv

load_dotenv(".env")
email_server = os.getenv("EMAIL_SERVER")
email_port = os.getenv("EMAIL_PORT")
sender_email = os.getenv("EMAIL_SENDER")
email_password = os.getenv("EMAIL_PASSWORD")


class EmailServer():

    def __init__(self):
        self.server = smtplib.SMTP_SSL(email_server, email_port, timeout=30)
        self.server.login(sender_email, email_password)
    
    def send(self, user_email: str, subject: str, body: str, attachment_path: str="None", timestamp: str="temp"):

        msg = MIMEMultipart()
        msg["From"] = Header(sender_email, "utf-8")
        msg["To"] = Header(user_email, "utf-8")
        msg["Subject"] = Header(subject, "utf-8")

        msg.attach(MIMEText(body, "plain", "utf-8"))

        if attachment_path != "None":
            with open(attachment_path, "rb") as f:
                part = MIMEApplication(f.read(), Name=os.path.basename(attachment_path))
                part["Content-Disposition"] = f'attachment; filename="{os.path.basename(attachment_path)}"'
                msg.attach(part)
        
        try:
            self.server.sendmail(sender_email, user_email, msg.as_string())
            self.server.quit()
            logging.info(f"[Email] Success! Email to {user_email}")
            return True
        except Exception as e:
            email_time = datetime.now().strftime('%Y%m%d_%H%M%S')
            err_log = f"[{timestamp}]\n[Email Time] {email_time}\n[To] {user_email}\n[Subject] {subject}\n[Attachment] {os.path.basename(attachment_path)}\n[Err] {e}\n\n"
            with open(f"./tmp/email_err_{timestamp}", 'a') as f:
                f.write(err_log)
            logging.info(f"[Email] Failed: {e}")
            return False


if __name__ == '__main__':

    email_server = EmailServer()

    user_email = " "
    subject = "test email"
    body = "test email from OpenBioMed"
    attachment_path = "./tmp/mol_2025_04_01_10_53_59_796.pdbqt"

    email_server.send(user_email=user_email, subject=subject, body=body, attachment_path=attachment_path)