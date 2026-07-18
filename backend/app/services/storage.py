import boto3
from botocore.client import Config
from app.core.config import settings

class StorageService:
    def __init__(self):
        # Configure Boto3 client to speak to MinIO or AWS S3
        self.s3 = boto3.client(
            "s3",
            endpoint_url=settings.S3_ENDPOINT,
            aws_access_key_id=settings.S3_ACCESS_KEY,
            aws_secret_access_key=settings.S3_SECRET_KEY,
            config=Config(signature_version="s3v4"),
            region_name="us-east-1"
        )
        self.bucket = settings.S3_BUCKET
        self._ensure_bucket_exists()

    def _ensure_bucket_exists(self):
        try:
            self.s3.head_bucket(Bucket=self.bucket)
        except Exception:
            try:
                self.s3.create_bucket(Bucket=self.bucket)
            except Exception as e:
                # In production environment, assume bucket exists or throws exception
                print(f"Warning: Could not create S3 bucket: {str(e)}")

    def upload_file(self, file_data: bytes, s3_key: str, content_type: str = "application/pdf") -> str:
        """
        Uploads raw file bytes to S3 object storage.
        """
        self.s3.put_object(
            Bucket=self.bucket,
            Key=s3_key,
            Body=file_data,
            ContentType=content_type
        )
        return s3_key

    def download_file(self, s3_key: str, dest_path: str):
        """
        Downloads a file from S3 to a local file path.
        """
        self.s3.download_file(
            Bucket=self.bucket,
            Key=s3_key,
            Filename=dest_path
        )

    def get_signed_url(self, s3_key: str, expiration: int = 3600) -> str:
        """
        Generates a temporary pre-signed URL to access the PDF document from the browser.
        """
        return self.s3.generate_presigned_url(
            "get_object",
            Params={"Bucket": self.bucket, "Key": s3_key},
            ExpiresIn=expiration
        )

storage_service = StorageService()
