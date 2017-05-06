#include <stdio.h>
#include <sys/time.h>
#include <stdlib.h>
#include <CL/cl.h>

#define MAX_SOURCE_SIZE 50000

cl_double get_time(cl_event event){
	cl_ulong queue = 0, complete = 0;
	clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_QUEUED, sizeof(cl_ulong), &queue, NULL);
	clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_COMPLETE, sizeof(cl_ulong), &complete, NULL);
	cl_double _time = (cl_double)(complete-queue)*(cl_double)(1e-06);
	return _time;
}

double calc_time(double t1,double t2){
        return (t2-t1);
}

double omp_get_time(){
        struct timeval T;
        gettimeofday(&T,NULL);
        return (T.tv_sec*1000+T.tv_usec/1000);
}


 
int main(int argc, char** argv){
double T1[5], T2[5];
double time_t[5];
size_t N = atoi(argv[1]);
size_t N2 = N/2;
double* M1 = malloc(sizeof(double)*N);
double* M2 = malloc(sizeof(double)*N2);
double *min = malloc(sizeof(double));
double *result = malloc(sizeof(double));
*result = 0;
int *semaphor = malloc(sizeof(int));

/*time*/
cl_double _time[5],min_time[5];
for (int j =0;j<5;++j){
	min_time[j] = 1000000;
}

/*props*/
cl_queue_properties props[] = {CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0};

cl_device_id device_id;
cl_context context;
cl_command_queue command_queue;
cl_program program;

/*for Generate stage*/ 
cl_kernel kernel_st1_1, kernel_st1_2;

/*For Map stage*/
cl_kernel kernel_st2_1, kernel_st2_2, kernel_st2_3;

/*For merge stage*/
cl_kernel kernel_st3;

/*For sort stage*/
cl_kernel kernel_st4_1, kernel_st4_2;

/*For reduce stage*/
cl_kernel kernel_st5_1, kernel_st5_2;

cl_platform_id platforms;
cl_uint num_devices;
cl_uint num_platforms;
cl_int clStatus;
cl_mem mem_M1, mem_M2,mem_Min, mem_Res, mem_Sem;
cl_event kernel_event[2];
cl_uint dims;
size_t local_size;
size_t single = 1;

FILE *fp;
char fileName[] = "./prog.cl";
char *source_str;
size_t source_size;
 
/* Load the source code containing the kernel*/
fp = fopen(fileName, "r");
if (!fp) {
fprintf(stderr, "Failed to load kernel.\n");
exit(1);
}

source_str = (char*)malloc(MAX_SOURCE_SIZE);
source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
fclose(fp);


/* start for */
for (int i=0;i<1;++i){

for (int j=0;j<5;++j)
	_time[j] = 0;

/*get platform*/
clGetPlatformIDs(1,&platforms,NULL);

/*get device*/
clGetDeviceIDs(platforms,CL_DEVICE_TYPE_CPU,1,&device_id,&num_devices);

clGetDeviceInfo(device_id, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS,sizeof(cl_uint), &dims, NULL);

clGetDeviceInfo(device_id, CL_DEVICE_MAX_WORK_GROUP_SIZE,sizeof(size_t), &local_size, NULL);

/*context for indetifing device*/
context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &clStatus);

/*create buffer for custom struct*/
mem_M1 = clCreateBuffer(context,CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, sizeof(double)*N,M1,&clStatus);
mem_M2 = clCreateBuffer(context,CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, sizeof(double)*N2,M2,&clStatus);
mem_Min = clCreateBuffer(context,CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, sizeof(double),min,&clStatus);
mem_Res = clCreateBuffer(context,CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, sizeof(double),result,&clStatus);
mem_Sem = clCreateBuffer(context,CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, sizeof(int),semaphor,&clStatus);

/*create program object*/
program = clCreateProgramWithSource(context, 1, (const char **)&source_str,(const size_t *)&source_size, &clStatus);

/*Build kernel program*/
clStatus = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);

/*Creating kernel object Generate stage*/
kernel_st1_1 = clCreateKernel(program, "gen1", &clStatus);
//kernel_st1_2 = clCreateKernel(program, "gen2", &clStatus);

/*Kernels for Map stage*/
kernel_st2_1 = clCreateKernel(program, "map1", &clStatus);
kernel_st2_2 = clCreateKernel(program, "map2", &clStatus);
kernel_st2_3 = clCreateKernel(program, "map3", &clStatus);

/*Kernels for Merge stage*/
kernel_st3 = clCreateKernel(program, "merge", &clStatus);

/*Kernel for Sort stage*/
kernel_st4_1 = clCreateKernel(program, "heap", &clStatus);
kernel_st4_2 = clCreateKernel(program, "sort", &clStatus);

/*Kernel for Reduce stage*/
kernel_st5_1 = clCreateKernel(program,"min_elem",&clStatus);
kernel_st5_2 = clCreateKernel(program,"reduce",&clStatus);

/**/
command_queue = clCreateCommandQueueWithProperties(context, device_id,props,&clStatus);


/* copy data to device */
/*clStatus= clEnqueueWriteBuffer(command_queue, NULL, CL_TRUE, 0, sizeof(double)*N, M1, 0, NULL, &kernel_event );
clStatus= clEnqueueWriteBuffer(command_queue, NULL, CL_TRUE, 0, sizeof(double)*N2, M2, 0, NULL, &kernel_event );
*/

/*Args for Generate stage*/
clSetKernelArg(kernel_st1_1, 0, sizeof(double)*N,(void *)&mem_M1);
clSetKernelArg(kernel_st1_1, 2, sizeof(size_t), &N);
clSetKernelArg(kernel_st1_1, 1, sizeof(double)*N2,(void *)&mem_M2);
clSetKernelArg(kernel_st1_1, 3, sizeof(size_t), &N2);

/*Args for Map stage*/
clSetKernelArg(kernel_st2_1, 0, sizeof(double)*N,(void *)&mem_M1);
clSetKernelArg(kernel_st2_1, 1, sizeof(size_t), &N2);
clSetKernelArg(kernel_st2_2, 0, sizeof(double)*N2,(void *)&mem_M2);
clSetKernelArg(kernel_st2_2, 1, sizeof(size_t), &N2);
clSetKernelArg(kernel_st2_3, 0, sizeof(double)*N2,(void *)&mem_M2);
clSetKernelArg(kernel_st2_3, 1, sizeof(size_t), &N2);

/*Args for Merge stage*/
clSetKernelArg(kernel_st3, 0, sizeof(double)*N,(void *)&mem_M1);
clSetKernelArg(kernel_st3, 1, sizeof(double)*N2,(void *)&mem_M2);
clSetKernelArg(kernel_st3, 2, sizeof(size_t), &N2);

/*Args for Sort stage*/
clSetKernelArg(kernel_st4_1, 0, sizeof(double)*N2,(void *)&mem_M2);
clSetKernelArg(kernel_st4_1, 1, sizeof(size_t),&N2);
clSetKernelArg(kernel_st4_2, 0, sizeof(double)*N2,(void *)&mem_M2);
clSetKernelArg(kernel_st4_2, 1, sizeof(size_t),&N2);

/*Args for Reduce stage*/
clSetKernelArg(kernel_st5_1, 0, sizeof(double)*N2,(void *)&mem_M2);
clSetKernelArg(kernel_st5_1, 1, sizeof(size_t),&N2);
clSetKernelArg(kernel_st5_1, 2, sizeof(double),&mem_Min);
clSetKernelArg(kernel_st5_2, 0, sizeof(double)*N2,(void *)&mem_M2);
clSetKernelArg(kernel_st5_2, 1, sizeof(size_t),&N2);
clSetKernelArg(kernel_st5_2, 2, sizeof(double),&mem_Min);
clSetKernelArg(kernel_st5_2, 3, sizeof(double),&mem_Res);
clSetKernelArg(kernel_st5_2, 4, sizeof(int),&mem_Sem);

/*Execute Generate stage */
clEnqueueNDRangeKernel(command_queue, kernel_st1_1, 1, NULL, &N, &local_size, 0,NULL, &kernel_event[0]);
//clEnqueueNDRangeKernel(command_queue, kernel_st1_2, 1, NULL, &N2, &local_size, 0,NULL, &kernel_event[1]);
//clWaitForEvents(1, &kernel_event[1]);
clWaitForEvents(1, &kernel_event[0]);
_time[0] = get_time(kernel_event[0]);

/*Execute Map stage */
clEnqueueNDRangeKernel(command_queue, kernel_st2_1, 1, NULL, &N, &local_size, 0,NULL, &kernel_event[0]);
clWaitForEvents(1, &kernel_event[0]);
_time[1] += get_time(kernel_event[0]);
clEnqueueNDRangeKernel(command_queue, kernel_st2_2, 1, NULL, &single, &single, 0,NULL, &kernel_event[0]);
clWaitForEvents(1, &kernel_event[0]);
_time[1] += get_time(kernel_event[0]);
clEnqueueNDRangeKernel(command_queue, kernel_st2_3, 1, NULL, &N2, &local_size, 0,NULL, &kernel_event[0]);
clWaitForEvents(1, &kernel_event[0]);
_time[1] += get_time(kernel_event[0]);

/*Execute Merge stage*/
clEnqueueNDRangeKernel(command_queue, kernel_st3, 1, NULL, &N2, &local_size, 0,NULL, &kernel_event[0]);
clWaitForEvents(1, &kernel_event[0]);
_time[2] = get_time(kernel_event[0]);

/*Execute Sort stage*/
size_t z = 4;
clEnqueueNDRangeKernel(command_queue, kernel_st4_1, 1, NULL, &z, &z, 0,NULL, &kernel_event[0]);
clWaitForEvents(1, &kernel_event[0]);
_time[3] += get_time(kernel_event[0]);
clEnqueueNDRangeKernel(command_queue, kernel_st4_2, 1, NULL, &single, &single, 0,NULL, &kernel_event[0]);
clWaitForEvents(1, &kernel_event[0]);
_time[3] += get_time(kernel_event[0]);

/*Execute Reduce stage*/
clEnqueueNDRangeKernel(command_queue, kernel_st5_1, 1, NULL, &single, &single, 0,NULL, &kernel_event[0]);
clWaitForEvents(1, &kernel_event[0]);
_time[4] += get_time(kernel_event[0]);
clEnqueueNDRangeKernel(command_queue, kernel_st5_2, 1, NULL, &N2, &local_size, 0,NULL, &kernel_event[0]);
clWaitForEvents(1, &kernel_event[0]);
_time[4] += get_time(kernel_event[0]);

for (int j=0;j<5;++j){
	if (_time[j] < min_time[j]){
		min_time[j] = _time[j];
	}
}

/*Clear*/
clReleaseMemObject(mem_M1);
clReleaseMemObject(mem_M2);
clReleaseMemObject(mem_Min);
clReleaseMemObject(mem_Res);
clReleaseMemObject(mem_Sem);
 
clReleaseKernel(kernel_st1_1);
//clReleaseKernel(kernel_st1_2);
clReleaseKernel(kernel_st2_1);
clReleaseKernel(kernel_st2_2);
clReleaseKernel(kernel_st2_3);
clReleaseKernel(kernel_st3);
clReleaseKernel(kernel_st4_1);
clReleaseKernel(kernel_st4_2);
clReleaseKernel(kernel_st5_1);
clReleaseKernel(kernel_st5_2);

clReleaseProgram(program);

clReleaseCommandQueue(command_queue);

clReleaseContext(context);
}//end of for
for (int j=0;j<5;++j){
	printf("%f ",_time[j]);
}
free(source_str);
free(M1);
free(M2);
free(result);
free(min);
return 0;
}
