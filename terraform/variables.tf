variable "region" {
  description = "AWS region. Keep us-east-1: the STScI public dataset mirror (s3://stpubdata) lives there, so in-region reads are free."
  type        = string
  default     = "us-east-1"
}

variable "project" {
  description = "Name prefix for resources."
  type        = string
  default     = "exoplanet-detector"
}
