using System;
using System.Collections.Generic;

namespace clrays
{
    public static class Info
    {
        public static List<(string, uint)> Textures = new List<(string, uint)>();
        public static uint MetaSize, SceneSize;
        public static uint IntMapSize, FloatMapSize;

        public static void PrintInfo()
        {
            Console.WriteLine($"Metadata: {MetaSize} B.");
            Console.WriteLine($"Scene:    {SceneSize} B.");
            Console.WriteLine($"Int FB:   {IntMapSize} B.");
            Console.WriteLine($"Float FB: {FloatMapSize} B.");
            uint sum = 0;
            for(int i = 0; i < Textures.Count; i++)
            {
                var name = Textures[i].Item1;
                var size = Textures[i].Item2;
                sum += size;
                Console.WriteLine($"Texture{i} : {name} : {size} B.");
            }
            Console.WriteLine("Totalsize: ");
            PrintSizeVerbose(sum);
            Console.WriteLine("Grand Total: ");
            sum += MetaSize + SceneSize + IntMapSize + FloatMapSize;
            PrintSizeVerbose(sum);
        }

        private static void PrintSizeVerbose(uint size)
        {
            Console.WriteLine($"    {size / Math.Pow(1024, 0)} B.");
            Console.WriteLine($"    {size / Math.Pow(1024, 1)} KB.");
            Console.WriteLine($"    {size / Math.Pow(1024, 2)} MB.");
            Console.WriteLine($"    {size / Math.Pow(1024, 3)} GB.");
        }
    }
}
