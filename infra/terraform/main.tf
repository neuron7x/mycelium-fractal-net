terraform {
  required_version = ">= 1.5.0"
  required_providers {
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = ">= 2.24.0"
    }
  }
}

provider "kubernetes" {
  config_path = var.kubeconfig_path
}

resource "kubernetes_namespace" "mfn" {
  metadata {
    name = var.namespace
    labels = {
      app = "mycelium-fractal-net"
      "pod-security.kubernetes.io/enforce" = "restricted"
      "pod-security.kubernetes.io/audit"   = "restricted"
      "pod-security.kubernetes.io/warn"    = "restricted"
    }
  }
}

resource "kubernetes_service_account" "mfn" {
  metadata {
    name      = "mfn-service-account"
    namespace = kubernetes_namespace.mfn.metadata[0].name
    labels = {
      app = "mycelium-fractal-net"
    }
  }
  automount_service_account_token = false
}

output "namespace" {
  description = "Namespace created for MyceliumFractalNet"
  value       = kubernetes_namespace.mfn.metadata[0].name
}
