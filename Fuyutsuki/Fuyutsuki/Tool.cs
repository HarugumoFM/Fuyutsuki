using System;
using System.Runtime.CompilerServices;
using System.Runtime.Serialization;

namespace Fuyutsuki
{
    [Serializable]
    public class Tool
    {
        public class Function:Matrix
        {
            public Matrix Data;
            public Matrix Gradient;
            
            Matrix Forward(Matrix num)
            {
                Data = num;
                return num;
            }
            Matrix Backward(Matrix num)
            {
                Gradient = num;
                return num;
            }
        }
        public class Square : Function
        {
            public Matrix Forward(Matrix num)
            {
                this.Data = num;
                return Mul(Data, Data);

            }
            public Matrix Backward(Matrix gx)
            {
                return Mul(Mul(gx, 2), gx);
            }
        }
        public class Exp : Function
        {
            public Matrix Forward(Matrix num)
            {
                this.Data = num;
                return Exp(num);
            }
            public Matrix BackWard(Matrix gy)
            {
                return Mul(Data, Mul(Data, gy));
            }
        }
    }
   
}
