using System.Globalization;
using System.Threading;
using FruckEngine;

namespace clrays {
    internal class Program {
        public static void Main(string[] args) { 
        Thread.CurrentThread.CurrentCulture = CultureInfo.GetCultureInfo( "en-US" );
            using (var win = new Window(1600, 900, "clrays", new Rays())) { win.Run(30.0, 60.0); }
        }
    }
}