using System;
using System.Collections.Generic;
using System.Data;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;
namespace Fuyutsuki
{
    public class DataLoader
    {
        Fuyutsuki.DataSets.DataSet dataset;
        int BatchSize;
        int DataSize;
        int MaxIter;
        int Iteration;
        bool Shuffle;
        int[] Index;
        public DataLoader(Fuyutsuki.DataSets.DataSet set,int batchSize, bool shuffle=true)
        {

            this.dataset = set;
            BatchSize = batchSize;
            Shuffle = shuffle;
            DataSize = dataset.Len();
            MaxIter = (int)Math.Ceiling((double)this.DataSize / BatchSize);

            this.Reset();

        }
        public void Reset()
        {
            this.Iteration = 0;
            if (this.Shuffle)
                this.Index = Permutation(this.dataset.Len());
            else
                this.Index = Arrange(this.dataset.Len());
                
        }
        public virtual void Iterator()
        {
            
        }
        public virtual (Matrix x,int[] t) Next()
        {
            if (Iteration >= MaxIter)
                Reset();
            int i = this.Iteration;
            var batchIndex= Index.Skip(i * BatchSize).Take(BatchSize).ToArray();
            (var x, var t) = dataset.GetItem(batchIndex);
            Iteration++;
            return (x, t);
        }

         int[] Permutation(int num)
        {
            var result = new int[num];
            for (int i = 0; i < num; i++)
            {
                result[i] = i;
            }
            return result.OrderBy(nt => Guid.NewGuid()).ToArray();
        }
        int[] Arrange(int num)
        {
            var result = new int[num];
            for (int i = 0; i < num; i++)
            {
                result[i] = i;
            }
            return result;
        }
    }
}
