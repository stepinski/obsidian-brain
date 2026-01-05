package main

import (
	"bytes"
	"io"
	"log"
	"mime/multipart"
	"net/http"
	"os"
	"path/filepath"

	"github.com/fsnotify/fsnotify"
)

const (
	inboxPath   = "../vault/inbox"
	labEndpoint = "http://localhost:8000/process" // Change to Lab IP
)

func main() {
	watcher, _ := fsnotify.NewWatcher()
	defer watcher.Close()

	os.MkdirAll(inboxPath, 0755)

	go func() {
		for {
			select {
			case event, ok := <-watcher.Events:
				if !ok {
					return
				}
				if event.Op&fsnotify.Create == fsnotify.Create {
					if filepath.Ext(event.Name) == ".pdf" {
						log.Printf("📂 New PDF: %s. Sending to Lab...", event.Name)
						sendToLab(event.Name)
					}
				}
			case err := <-watcher.Errors:
				log.Println("error:", err)
			}
		}
	}()

	watcher.Add(inboxPath)
	log.Printf("🚀 Watching for notes in %s", inboxPath)
	<-make(chan struct{})
}

func sendToLab(path string) {
	file, _ := os.Open(path)
	defer file.Close()

	body := &bytes.Buffer{}
	writer := multipart.NewWriter(body)
	part, _ := writer.CreateFormFile("file", filepath.Base(path))
	io.Copy(part, file)
	writer.Close()

	http.Post(labEndpoint, writer.FormDataContentType(), body)
}
