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

resource "kubernetes_limit_range" "mfn" {
  metadata {
    name      = "mfn-limit-range"
    namespace = kubernetes_namespace.mfn.metadata[0].name
  }
  spec {
    limit {
      type = "Container"
      default_request = {
        cpu    = "250m"
        memory = "256Mi"
      }
      default = {
        cpu    = "500m"
        memory = "512Mi"
      }
    }
  }
}

resource "kubernetes_resource_quota" "mfn" {
  metadata {
    name      = "mfn-resource-quota"
    namespace = kubernetes_namespace.mfn.metadata[0].name
  }
  spec {
    hard = {
      pods                     = "150"
      "requests.cpu"           = "30"
      "limits.cpu"             = "60"
      "requests.memory"        = "32Gi"
      "limits.memory"          = "48Gi"
      "requests.ephemeral-storage" = "30Gi"
      "limits.ephemeral-storage"   = "60Gi"
    }
  }
}

resource "kubernetes_network_policy" "mfn_restricted" {
  metadata {
    name      = "mfn-restricted"
    namespace = kubernetes_namespace.mfn.metadata[0].name
  }
  spec {
    pod_selector {
      match_labels = {
        app = "mycelium-fractal-net"
      }
    }
    policy_types = ["Ingress", "Egress"]
    ingress {
      from {
        pod_selector {}
      }
    }
    egress {
      to {
        pod_selector {}
      }
    }
    egress {
      to {
        namespace_selector {
          match_labels = {
            "kubernetes.io/metadata.name" = "kube-system"
          }
        }
      }
      ports {
        port     = 53
        protocol = "UDP"
      }
      ports {
        port     = 53
        protocol = "TCP"
      }
    }
  }
}

resource "kubernetes_pod_disruption_budget_v1" "mfn" {
  metadata {
    name      = "mfn-pdb"
    namespace = kubernetes_namespace.mfn.metadata[0].name
  }
  spec {
    min_available = 2
    selector {
      match_labels = {
        app = "mycelium-fractal-net"
      }
    }
  }
}

output "namespace" {
  description = "Namespace created for MyceliumFractalNet"
  value       = kubernetes_namespace.mfn.metadata[0].name
}
