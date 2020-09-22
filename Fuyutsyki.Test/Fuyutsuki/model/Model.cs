using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Fuyutsuki.model;
using Fuyutsuki.tool;
using static Fuyutsuki.tool.Activations;
using static Fuyutsuki.model.Layers;
using System.Reflection.Metadata.Ecma335;

namespace Fuyutsuki.model
{
    [Serializable]
    public class Model : Layers.Layer
    {

    }
    [Serializable]
    public class TwoLayerNetwork : Model
    {
        public Layers.Layer l1;
        public Layers.Layer l2;
        public TwoLayerNetwork(int hiddenSize, int outSize)
        {

            this.l1 = new Layers.Linear(hiddenSize);
            this.l2 = new Layers.Linear(outSize);
            Layers.Add(l1);
            Layers.Add(l2);
        }
        public override Variable Forward(Variable inputs)
        {

            var y = F.Sigmoid(l1.Forward(inputs));
            y = l2.Forward(y);
            return y;

        }
    }

    /// <summary>
    /// 多層パーセプトロン
    /// </summary>
    [Serializable]
    public class MLP : Model
    {
        Activation activation;
        /// <summary>
        /// 
        /// </summary>
        /// <param name="fcOutputSizes">各層のサイズのリスト</param>
        public MLP(List<int> fcOutputSizes,Activation act)
        {
            activation = act;
            foreach (int outSize in fcOutputSizes)
            {
                var layer = new Layers.Linear(outSize);
                Layers.Add(layer);
            }
        }
        public MLP(List<int> fcOutputSizes)
        {
            foreach (int outSize in fcOutputSizes)
            {
                activation = new Sigmoid();
                var layer = new Layers.Linear(outSize);
                Layers.Add(layer);
            }
        }
        public override Variable Forward(Variable inputs)
        {
            if (Layers.Count > 1)
            {
                var y = activation.Forward(Layers[0].Forward(inputs));
                for (int i = 1; i < Layers.Count - 1; i++)
                {
                    y = activation.Forward(Layers[i].Forward(y));
                }
                return Layers[Layers.Count - 1].Forward(y);
            }
            else
                return Layers[0].Forward(inputs);
            
        }
    }

    /* リカレントニューラルネットワーク*/
    [Serializable]
    public class SimpleRNN:Model
    {
        GRUCell rnn;
        Layer fc;

        public SimpleRNN(int hiddenSize, int outSize)
        {
            this.rnn = new GRUCell(hiddenSize);
            this.fc = new Linear(outSize);
            Layers.Add(rnn);
            Layers.Add(fc);


        }
        public void ResetState()
        {
            rnn.ResetState();
        }
        public override Variable Forward(Variable inputs)
        {
            var h = rnn.Forward(inputs);
            var y = fc.Forward(h);
            return y;
        }
    }

    [Serializable]
    public class Encoder:Model
    {
        GRU gru;
        List<Linear> Vec2Hiddens = new List<Linear>();
        int Depth;//GRUセルの深さ
        int EmbeddDim;//文字の埋め込み次元
        int HiddenDim;//隠れ層のサイズ
        public Encoder( int embeddDim,int hiddenDim, int depth=1,bool dropout=false)
        {
            gru = new GRU(hiddenDim,embeddDim,3,dropout);
            Layers.Add(gru);
            this.Depth = depth;
            this.EmbeddDim = embeddDim;
            this.HiddenDim = hiddenDim;
        }
        public (Variable,List<Variable>) Forward(List<Variable> inputs, bool Train = false)
        {
           
           (var state,var outputs)=gru.Forward(inputs);
            
            return (state,outputs);
        }
        public void RecetStates()
        {
            
            gru.ResetStatu();
        }
    }

    [Serializable]
    public class Decoder : Model
    {
        //Neural Network
        public GRU gru;
        public Linear Hidden2Layers;
        public Embedding embedding;

        //member
        int EmbeddDim;//文字の埋め込み次元
        int HiddenDim;//隠れ層のサイズ
        int VocabSize;

        public Decoder(int vocabSize, int embeddDim, int hiddenDim,int depth=1)
        {
            
          
            this.EmbeddDim = embeddDim;
            this.HiddenDim = hiddenDim;
            this.VocabSize = vocabSize;
            gru = new GRU(hiddenDim,embeddDim,depth,true);
            Layers.Add(gru);
            embedding = new Embedding(vocabSize, embeddDim);
           
                this.Hidden2Layers= new Linear(hiddenDim, VocabSize);
                Layers.Add(Hidden2Layers);
            
        }
        public List<Variable> Forward(List<Variable>  inputs,Variable EncorderState)
        {
            
              (Variable state, List<Variable> output) = gru.Forward(inputs,EncorderState,true);
            //全結合層でvocabsizeに成形
            for(int i=0;i<inputs.Count;i++)
                output[i] = Hidden2Layers.Forward(output[i]);
            
            return output;
        }
     


        public void ResetStates()
        {
            gru.ResetStatu();
            
        }
        
    }

    
}
