SCRIPTS_DIR := conilab/scripts_to_run

# Target directory for symlinks
INSTALL_DIR := ~/.conilab
export PATH="$HOME/~conilab:$PATH"
# Find all Python scripts
SCRIPTS := $(wildcard $(SCRIPTS_DIR)/*.py)

#SCRIPTS := $(patsubst %.py, %, $(notdir $(wildcard $(SCRIPTS_DIR)/*.py)))

.PHONY: install

# Create symlinks for all scripts
# Install the scripts into the INSTALL_DIR
.PHONY: install
install:
	@/bin/mkdir -p $(INSTALL_DIR)
	@for script in $(SCRIPTS); do \
		/bin/cp $$script $(INSTALL_DIR)/$$(/usr/bin/basename $$script .py); \
		/bin/chmod +x $(INSTALL_DIR)/$$(/usr/bin/basename $$script .py); \
	done
	@echo "Scripts installed to $(INSTALL_DIR)"
	@if ! /usr/bin/grep -q 'export PATH=.*conilab' ~/.zshrc; then \
		echo 'export PATH="$$HOME/.conilab:$$PATH"' >> ~/.zshrc; \
	fi

# Uninstall the scripts
.PHONY: uninstall
uninstall:
	@for script in $(SCRIPTS); do \
		/bin/rm -f $(INSTALL_DIR)/$$script; \
	done
	@echo "Scripts uninstalled from $(INSTALL_DIR)"
	@/usr/bin/sed -i '' '/export PATH="\$$HOME\/.conilab:\$$PATH"/d' ~/.zshrc

# Default action is to run install
all: install