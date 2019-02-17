// #define GLINTEROP
#ifdef GLINTEROP
__kernel void render(
    write_only image2d_t image_buffer
) {
#else
__kernel void render(
    __global int *image_buffer,
    uint w,
    uint h
) {
#endif
    int x = get_global_id(0);
    int y = get_global_id(1);
    uint id = x + y * w;
    
    float u = (float)x / w;
    float v = (float)y / h;
    float2 uv = (float2)(u - 0.5f, v - 0.5f);
    
    int r = u * 255;
    int g = v * 255;
    int b = 0;
    int fres = (r << 16) + (g << 8) + b;
    
#ifdef GLINTEROP
    int2 pos = (int2)(x, y);
	write_imagef(image_buffer, pos, fres);
#else
    image_buffer[id] = fres;
#endif
}