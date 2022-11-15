#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

int main(int argc, char* agrc[])
{
	pid_t pid;

	pid = fork();
	printf("Hello world from pid: %u\n", pid);
	return (0);
}
