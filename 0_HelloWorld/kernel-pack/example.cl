/*
* example.cl
*
*  date    : 01.04.2020
*  author  : M. Khaled
*/

#define CFG_MSG @pfaces-configValueString:"config_message"
#define PARAM_MSG @@param_message@@

kernel void exampleKernelFunction(global char* data_in){
    printf("Hello World from the device side!\n");
    printf("The value of the config item (config_message) is: %s\n", CFG_MSG);
    printf("The value of the kernel param (param_message) is: %s\n", PARAM_MSG);

    if (get_global_id(0) == 0 && get_global_id(1) == 0) {
        data_in[0] = 'p';
        data_in[1] = 'F';
        data_in[2] = 'c';
        data_in[3] = 's';
    }
}
