#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>

int main(void)
{
	pid_t child_pid;

	child_pid = fork();
	if (child_pid == -1)
	{
	perror("Error;");
	return (1);
	}
	if (child_pid == 0)
	{
//		sleep(3);
		printf("this is the child process\n");
	}
	else
	{
		sleep(3);
		printf("this is the parent process\n");
	}
	return (0);
}
