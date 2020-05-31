using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Fuyutsuki
{
    [Serializable]
    public class Matrix
    {
        public int Row;
        public int Column;
        public double[,] Data;
        public Matrix()
        { }
        public Matrix(double[,] data)
        {
            Data = data;
            Row = data.GetLength(0);
            Column = data.GetLength(1);

        }
        public Matrix(double data)
        {
            Data = new double[1, 1] { { data } };

            Row = 1;
            Column = 1;

        }
        public Matrix Mul(Matrix a, Matrix b)
        {
            int n = a.Column;
            var result = CreateMatrix(a.Row, b.Column);
            for (int i = 0; i < a.Row; i++)
            {
                for (int j = 0; j < b.Column; j++)
                {
                    double sum = 0;
                    for (int k = 0; k < n; k++)
                    {
                        sum += a.Data[i, k] * b.Data[k, j];
                    }
                    result.Data[i, j] = sum;

                }
            }
            return result;
        }
        public Matrix Mul(Matrix a, double b)
        {
            int n = a.Column;
            var result = CreateMatrix(a.Row, a.Column);
            for (int i = 0; i < a.Row; i++)
            {
                for (int j = 0; j < a.Column; j++)
                {
                    result.Data[i, j] *= b;

                }
            }
            return result;
        }
        public Matrix CreateMatrix(int row, int column)
        {
            Matrix result = new Matrix(new double[row, column]);
            return result;


        }

        public Matrix Exp(Matrix a)
        {
            var result = CreateMatrix(a.Row, a.Column);
            for (int i = 0; i < a.Row; i++)
            {
                for(int j=0;j<a.Column;j++)
                {
                    result.Data[i, j] = Math.Exp(a.Data[i, j]);
                }
            }
            return result;
        }
    }
}
