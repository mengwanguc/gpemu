package main

import (
	"fmt"
	"os"
	"log"
	"path/filepath"
	"strings"
)

type EGPU struct {
	HostPath      string `json:"hostPath"`
	ContainerPath string `json:"containerPath"`
	Permission    string `json:"permission"`
}

func (d *EGPU) validate() error {
	numGlobHostPath := strings.Count(d.HostPath, "*")
	if numGlobHostPath > 1 {
		return fmt.Errorf("HostPath can container only one '*' character: %s", d.HostPath)
	}

	if numGlobHostPath == 1 {
		if !strings.HasSuffix(d.ContainerPath, "*") {
			return fmt.Errorf("ContainerPath should ends with '*' character when HostPath container '*': %s", d.ContainerPath)
		}
		return nil
	}

	if strings.Contains(d.ContainerPath, "*") {
		return fmt.Errorf("ContainerPath must not contain '*' when HostPath does not contain '*': %s", d.ContainerPath)
	}

	return nil
}

type ExpandedEGPU struct {
	HostPath      string
	ContainerPath string
	Permission    string
}

func (d EGPU) Expand() ([]*ExpandedEGPU, error) {
	if err := d.validate(); err != nil {
		return nil, err
	}

	log.Println("d.HostPath: ", d.HostPath)
    
	matchedHostPath, err := filepath.Glob(d.HostPath)

	log.Println("matchedHostPath ", matchedHostPath)

	if err != nil {
		return nil, err
	}

	expanded := []*ExpandedEGPU{}
	baseHostPath := strings.Split(d.HostPath, "*")[0]
	baseContainerPath := strings.Split(d.ContainerPath, "*")[0]
	for _, hp := range matchedHostPath {
		fInfo, _ := os.Stat(hp)
		if fInfo.IsDir() {
			continue
		}

		expanded = append(expanded, &ExpandedEGPU{
			HostPath:      hp,
			ContainerPath: strings.Replace(hp, baseHostPath, baseContainerPath, 1),
			Permission:    d.Permission,
		})
	}
	return expanded, nil
}
