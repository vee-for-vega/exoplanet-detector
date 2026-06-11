output "bucket_name" {
  description = "Data bucket name. Export as EXOPLANET_S3_BUCKET for src/data/s3_sync.py."
  value       = aws_s3_bucket.data.bucket
}
