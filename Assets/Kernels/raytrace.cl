// #define GLINTEROP
#ifdef GLINTEROP
__kernel void render(
    write_only image2d_t image_buffer
) {
#else
__kernel void render(
    __global uchar *image_buffer
) {
#endif
    int x = get_global_id(0);
    int y = get_global_id(1);
    uint id = x + y * 1600;
    
#ifdef GLINTEROP
    int2 pos = (int2)(x, y);
	write_imagef(image_buffer, pos, 0xFF);
#else
    image_buffer[id] = 0xFF;
#endif
}