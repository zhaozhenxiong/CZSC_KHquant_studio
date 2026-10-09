group "default" { targets = ["khquant-czsc"] }
target "khquant-czsc" {
  context = "."
  dockerfile = "Dockerfile"
  tags = ["khquant:czsc-local"]
}
