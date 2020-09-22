using System;
using System.Collections.Generic;
using Fuyutsuki.model;
using static Fuyutsuki.Matrix;

namespace Fuyutsuki.tool
{
    public　class Optimizer
    {
        public Model target;
        public List<Optimizer> Hooks = new List<Optimizer>();
        public void SetUp(Model model)
        {
            this.target = model;
        }
        public virtual void Update()
        {
            List<Parameter> Params = new List<Parameter>();
            foreach (var param in target.GetParam())
            {
                if (param.Grads != null)
                    Params.Add(param);
                
            }
            foreach (var param in Params)
            {
                this.UpdateOne(param);
            }
        }
        public virtual void UpdateOne(Parameter param)
        {
            throw new NotImplementedException();
        }
        public void  AddHook(Optimizer f)
        {
            this.Hooks.Add(f);
        }
        
        
    }

    public class Optimizers
    {
        public class SGD : Optimizer
        {
            double lr;
            public SGD(double lr = 0.01)
            {
                this.lr = lr;
            }
            public override void UpdateOne(Parameter param)
            {
                param.Weights -= lr * param.Grads;
            }
        }

        public class Adam:Optimizer
        {
            Dictionary<Parameter, Matrix> ms;
            Dictionary<Parameter, Matrix> vs;
            double T;
            double Alpha;
            double Beta1;
            double Beta2;
            double Eps;
            public Adam(double alpha=0.001,double beta1=0.9,double beta2=0.999,double eps=1e-8)
            {
                this.T = 0;
                this.Alpha = alpha;
                this.Beta1 = beta1;
                this.Beta2 = beta2;
                this.Eps = eps;
                ms = new Dictionary<Parameter, Matrix>();
                vs = new Dictionary<Parameter, Matrix>();

            }
            public override void Update()
            {
                this.T+=1;
                base.Update();
            }
            public double Lr()
            {
                var fix1=1.0 - Math.Pow(Beta1, T);
                var fix2 = 1.0 - Math.Pow(Beta2, T);
                return Alpha * Math.Sqrt(fix2) / fix1;
            }
            public override void UpdateOne(Parameter param)
            {
                if (!ms.ContainsKey(param))
                {
                    ms.Add(param,ZerosLike(param.Weights));
                    vs.Add(param, ZerosLike(param.Weights));
                }
                var m = ms[param];
                var v = vs[param];
                m += (1.0 - Beta1) * (param.Grads - m);
                v += (1.0 - Beta2) * (param.Grads*param.Grads - v);
                ms[param] = m;
                vs[param] = v;
                var x = Lr() * m;
                var y = Matrix.Sqrt(v) + Eps;
                param.Weights -=x/y;

                
            }
        }

    }
    
  

}
