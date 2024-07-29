FROM golang:1.21 as build
ARG CGO_ENABLED=0
ARG GOOS=linux
ARG GOARCH=amd64

WORKDIR /go/src/gpemu-k8s-device-plugin
COPY go.mod go.sum ./
RUN go mod download

COPY . .
RUN go install -ldflags="-s -w"

FROM gcr.io/distroless/static-debian11
COPY --from=build /go/bin/gpemu-k8s-device-plugin /bin/gpemu-k8s-device-plugin

CMD ["/bin/gpemu-k8s-device-plugin"]
