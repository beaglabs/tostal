terraform {
  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 4.0"
    }
  }
}

provider "azurerm" {
  features {}
}

resource "azurerm_resource_group" "tostal" {
  name     = var.resource_group_name
  location = var.location
}

resource "azurerm_storage_account" "tostal" {
  name                     = var.storage_account_name
  resource_group_name      = azurerm_resource_group.tostal.name
  location                 = azurerm_resource_group.tostal.location
  account_tier             = "Standard"
  account_replication_type = "LRS"
  account_kind             = "StorageV2"
}

resource "azurerm_storage_container" "models" {
  name                  = "container-models"
  storage_account_name  = azurerm_storage_account.tostal.name
  container_access_type = "private"
}