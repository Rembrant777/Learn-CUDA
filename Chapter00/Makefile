PROJECTS ?= $(shell find $(shell ls -d ./*) -name Makefile)

.PHONY: print_info_and_exit

print_and_exit:
	@echo $(PROJECTS)
	$(info Exiting after printing PROJECTS)

all: print_and_exit

