# S3 storage for the exoplanet detection pipeline.
#
# One private, versioned bucket holding everything except raw light curves:
#   metadata/   TCE tables + labels (TESS TOI, Kepler DR25)
#   processed/  phase-folded tensors (64x64 images, 1D signals)
#   models/     trained model checkpoints
#   results/    metrics CSVs and comparisons
#
# Raw light curves are deliberately NOT stored here. They already live in
# STScI's public mirror (s3://stpubdata/kepler/public, requester-pays,
# us-east-1); preprocessing streams them and keeps only the small tensors.
#
# Usage:
#   terraform init
#   terraform plan
#   terraform apply
#   export EXOPLANET_S3_BUCKET=$(terraform output -raw bucket_name)

terraform {
  required_version = ">= 1.5"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = ">= 5.0"
    }
  }
}

provider "aws" {
  region = var.region
}

data "aws_caller_identity" "current" {}

locals {
  bucket_name = "${var.project}-data-${data.aws_caller_identity.current.account_id}"
}

resource "aws_s3_bucket" "data" {
  bucket = local.bucket_name

  tags = {
    Project   = var.project
    ManagedBy = "terraform"
  }
}

resource "aws_s3_bucket_versioning" "data" {
  bucket = aws_s3_bucket.data.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_public_access_block" "data" {
  bucket = aws_s3_bucket.data.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_server_side_encryption_configuration" "data" {
  bucket = aws_s3_bucket.data.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

# Old object versions are kept 90 days, then expired, so iterating on
# processed tensors does not accumulate storage cost forever.
resource "aws_s3_bucket_lifecycle_configuration" "data" {
  bucket = aws_s3_bucket.data.id

  rule {
    id     = "expire-noncurrent-versions"
    status = "Enabled"

    filter {}

    noncurrent_version_expiration {
      noncurrent_days = 90
    }
  }
}
