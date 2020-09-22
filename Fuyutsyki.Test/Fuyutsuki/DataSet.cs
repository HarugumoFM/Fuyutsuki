using System;
using System.Collections.Generic;
using System.Linq;
using System.Security.Cryptography.X509Certificates;
using System.Text;
using System.Threading.Tasks;
using static Fuyutsuki.Matrix;

namespace Fuyutsuki
{
    public class DataSets
    {
        public class DataSet
        {
            public bool Train;
            public Matrix Data;
            public int[] Label;
            public DataSet(bool train = true)
            {
                Train = train;
                this.Prepare();
            }
            public virtual void Prepare()
            {
                
            }
            /// <summary>
            /// n番目のデータセットを出力
            /// </summary>
            /// <param name="n"></param>
            /// <returns></returns>
            public (Matrix data, int? label) GetItem(int n)
            {
                Matrix data=new Matrix(1,Data.Column);
                if (this.Label == null)
                {
                    for (int i = 0; i < Data.Column; i++)
                    {
                        data.Data[0, i] = this.Data.Data[n, i];
                    }
                    return (data, null);

                }
                else
                {
                    for (int i = 0; i < Data.Column; i++)
                    {
                        data.Data[0, i] = this.Data.Data[n, i];
                    }
                    return (data, Label[n]);
                }
                    
            }
            /// <summary>
            /// 複数の指定されたデータセットを取り出す
            /// </summary>
            /// <param name="n"></param>
            /// <returns></returns>
            public (Matrix data, int[] label) GetItem(int[] n)
            {
                Matrix data;
                if (this.Label == null)
                {
                    data = Data.Squeeze(n);
                    return (data, null);

                }
                else
                {
                    data = Data.Squeeze(n);
                    var label = Squeeze(Label, n);
                    return (data, label);
                }

            }
            /// <summary>
            /// データセットの大きさを出力
            /// </summary>
            /// <returns></returns>
            public int Len()
            {
                return Data.Row;
            }
            static int[] Squeeze(int[] x, int[] y)
            {
                var result = new int[y.Length];
                for (int i = 0; i < y.Length; i++)
                {
                    result[i] = x[y[i]];
                }
                return result;
            }

        }
        public class Spiral:DataSet
        {
            public Spiral(bool train=true)
            {
                this.Train = train;
                this.Prepare();
            }
            public override void Prepare()
            {
                (this.Data, this.Label) = GetSpiral(this.Train);
            }
        }
        public class DataSetMat
        {
            public bool Train;
            public Matrix[] Data;
            public int[] Label;
            public DataSetMat(bool train = true)
            {
                Train = train;
                this.Prepare();
            }
            public virtual void Prepare()
            {

            }
            /// <summary>
            /// n番目のデータセットを出力
            /// </summary>
            /// <param name="n"></param>
            /// <returns></returns>
            public (Matrix data, int? label) GetItem(int n)
            {
                Matrix data;
                if (this.Label == null)
                {
                    data = Data[n];
                    return (data, null);

                }
                else
                {
                    data = Data[n];
                    return (data, Label[n]);
                }

            }
            /// <summary>
            /// 複数の指定されたデータセットを取り出す
            /// </summary>
            /// <param name="n"></param>
            /// <returns></returns>
            public (Matrix[] data, int[] label) GetItem(int[] n)
            {
                Matrix[] data;
                if (this.Label == null)
                {
                    data = Squeeze(Data,n);
                    return (data, null);

                }
                else
                {
                    data = Squeeze(Data,n);
                    var label = Squeeze(Label, n);
                    return (data, label);
                }

            }
            /// <summary>
            /// データセットの大きさを出力
            /// </summary>
            /// <returns></returns>
            public int Len()
            {
                return Data.Length;
            }
            

        }


        public static int[] Squeeze(int[] x, int[] y)
        {
            var result = new int[y.Length];
            for (int i = 0; i < y.Length; i++)
            {
                result[i] = x[y[i]];
            }
            return result;
        }
        public static Matrix[] Squeeze(Matrix[] x, int[] y)
        {
            var result = new Matrix[y.Length];
            for (int i = 0; i < y.Length; i++)
            {
                result[i] = x[y[i]];
            }
            return result;
        }
        public static (Matrix x,int[] t) GetSpiral(bool train = true)
        {
            var rand = new Random(1984);
            int numData = 100;
            int numClass = 3;
            int inputDim = 2;
            int dataSize = numClass * numData;
            var x = CreateMatrix(dataSize, inputDim);
            var t = new int[dataSize] ;
            for (int j = 0; j < numClass; j++)
            {
                for (int i = 0; i < numData; i++)
                {
                    double rate = (double)i / numData;
                    var radius = 1.0 * rate;
                    var theta = j * 4.0 + 4.0 * rate + rand.NextDouble() * 0.2;
                    int ix = numData*j+i;
                    x.Data[ix, 0] = radius * Math.Sin(theta);
                    x.Data[ix, 1] = radius * Math.Cos(theta);
                    t[ix]=j;
                    
                }
            }
            int n = dataSize;
            return (x, t);



        }
    }
}
