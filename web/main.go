package main

import (
	"log"
	"net/http"
	"os"
	"time"
)

func main() {
	config, err := loadConfig(os.LookupEnv)
	if err != nil {
		log.Fatal(err)
	}
	app, err := newServer(config, newKYCClient(config, newHTTPClient().standard()), newSessionStore())
	if err != nil {
		log.Fatal(err)
	}
	server := &http.Server{
		Addr:              config.Address,
		Handler:           app.routes(),
		ReadHeaderTimeout: 5 * time.Second,
		ReadTimeout:       45 * time.Second,
		WriteTimeout:      50 * time.Second,
		IdleTimeout:       60 * time.Second,
	}
	log.Printf("KYC operator console listening on %s", config.Address)
	log.Fatal(server.ListenAndServe())
}
