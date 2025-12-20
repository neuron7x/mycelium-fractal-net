variable "namespace" {
  description = "Namespace to create for MyceliumFractalNet"
  type        = string
  default     = "mycelium-fractal-net"
}

variable "kubeconfig_path" {
  description = "Path to kubeconfig with access to the target cluster"
  type        = string
  default     = "~/.kube/config"
}
