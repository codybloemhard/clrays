struct RayHit{
    float3 pos;
    float3 nor;
    float t;
};

struct Ray{
    float3 pos;
    float3 dir;
};

struct RayHit InterSphere(struct Ray* r, float3 spos, float srad){
    struct RayHit hit;
    hit.t = -1.0;
    float3 l = spos - r->pos;
    float tca = dot(r->dir, l);
    float d = tca*tca - dot(l, l) + srad*srad;
    if(d < 0) return hit;
    float t = tca - sqrt(d);
    if(t < 0){
        t = tca + sqrt(d);
        if(t < 0) return hit;
    }
    hit.t = t;
    hit.pos = r->pos + r->dir * t;
    hit.nor = (hit.pos - spos) / srad;
    return hit;
}

#ifdef GLINTEROP
__kernel void render(
    write_only image2d_t image_buffer,
#else
__kernel void render(
    __global int *image_buffer,
#endif
    __global const float *sc_spheres,
    const uint w,
    const uint h,
    const uint sc_spheres_count
){
    int x = get_global_id(0);
    int y = get_global_id(1);
    uint pixid = x + y * w;
    
    float2 uv = (float2)(((float)x / w) - 0.5f, ((float)y / h) - 0.5f);
    uv.x *= ((float)w/h);
    
    char r = 0;
    char g = 0;
    char b = 0;
    
    struct Ray ray;
    ray.pos = (float3)(0,0,0);
    ray.dir = normalize((float3)(uv.x,uv.y,-1) - ray.pos);
    int i = 0;
    float3 spos = (float3)(sc_spheres[i + 0], sc_spheres[i + 1], sc_spheres[i + 2]);
    float srad = sc_spheres[i + 3];
    struct RayHit hit = InterSphere(&ray, spos, srad);
    r = max(0.0f, hit.t) * 50;
    
    int fres = (r << 16) + (g << 8) + b;

#ifdef GLINTEROP
    int2 pos = (int2)(x, y);
	write_imagef(image_buffer, pos, fres);
#else
    image_buffer[pixid] = fres;
#endif
}