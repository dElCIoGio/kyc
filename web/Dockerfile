# syntax=docker/dockerfile:1.7
FROM node:22-bookworm-slim AS assets

WORKDIR /src/web

COPY web/package.json ./
RUN npm install --ignore-scripts

COPY web/tailwind.config.js ./
COPY web/templates ./templates
COPY web/static ./static
RUN npm run build:assets

FROM golang:1.23-bookworm AS build

WORKDIR /src/web

COPY web/go.mod ./
RUN go mod download

COPY web/*.go ./
COPY web/templates ./templates
COPY web/static ./static
COPY --from=assets /src/web/static/app.css ./static/app.css
COPY --from=assets /src/web/static/htmx.min.js ./static/htmx.min.js

RUN CGO_ENABLED=0 GOOS=linux go build -trimpath -ldflags="-s -w" -o /out/kyc-operator-console .

FROM gcr.io/distroless/base-debian12:nonroot

WORKDIR /app
COPY --from=build --chown=nonroot:nonroot /out/kyc-operator-console ./kyc-operator-console

USER nonroot:nonroot
EXPOSE 8080

ENTRYPOINT ["/app/kyc-operator-console"]
