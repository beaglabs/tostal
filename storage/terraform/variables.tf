variable "resource_group_name" {
  description = "Azure resource group name"
  default     = "tostal-prod"
}

variable "location" {
  description = "Azure region"
  default     = "eastus2"
}

variable "storage_account_name" {
  description = "Azure Storage account name (must be globally unique, lowercase)"
  default     = "murmurativeprod"
}

variable "container_app_environment" {
  description = "Container Apps Environment name"
  default     = "tostal-env"
}