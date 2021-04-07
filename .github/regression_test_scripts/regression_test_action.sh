#!/bin/sh -l

do_a_test_expect_success()
{
	echo
	printf "\t \033[1;34m***> Running Test %s with command: %s\n\033[0m" "$2" "$1"
	echo

	# Run the command, parameter 1
	if ! $1;
	then
		echo
		printf "\t \033[1;31m***> Failed to %s\n\033[0m" "$2"
		echo
		exit 1
	else
		echo
		printf "\t \033[1;32m***> %s Completed, Success\n\033[0m" "$2"
		echo
	fi
}

do_a_test_expect_failure()
{
	echo
	printf "\t \033[1;34m***> Running Test %s with command: %s\n\033[0m" "$2" "$1"
	echo

	# Run the command, parameter 1

	if $1;
	then
		echo
		printf "\t \033[1;31m***> %s Completed without Error, but was expected to fail\n\033[0m" "$2"
		echo
		exit 1
	else
		echo
		printf "\t \033[1;32m***> %s Failed as expected, Success\n\033[0m" "$2"
		echo
	fi
}

show_banner()
{
    printf "\n\n \033[1;34m***********************************\n"
    printf     "  \033[1;34m%s" "$1"
    printf "\n\n \033[1;34m***********************************\n"
}

show_banner "Starting regression actions"

show_banner "Updating VM and install pre-requirements"
do_a_test_expect_success "uname -a" "Running on OS:"
do_a_test_expect_success "sudo apt-get update" "Update apt cache"
do_a_test_expect_success "sudo apt-get install -y python3-pip libsndfile1-dev" "Install required packages"
do_a_test_expect_success "export PATH="$PATH:/home/runner/.local/bin"" "Update path for pip installs"

do_a_test_expect_success "command -v python3" "python3 install check"
do_a_test_expect_success "command -v pip3" "pip3 install check"
do_a_test_expect_success "python3 -m pip install -U pip wheel setuptools" "Update python basics"
do_a_test_expect_success "python3 -m pip --version" "Show pip version"

do_a_test_expect_success "pwd" "Current directory"
do_a_test_expect_success "cd $GITHUB_WORKSPACE" "Changing to checkout directory..."
do_a_test_expect_success "pwd" "Checkout directory"
do_a_test_expect_success "ls" "Directory contents"

show_banner "Starting Regression Tests..."
do_a_test_expect_success "python3 -m pip install -r requirements.txt" "Install requirements.txt"
do_a_test_expect_success "pytest test_*.py" "Run all python tests of format test_*py"
