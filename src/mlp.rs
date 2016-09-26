pub trait NeuralNode
{
    fn feedforward(&mut self, values: &Vec<f32>) -> f32;
}

pub struct MLP<N: NeuralNode>
{
    layers: Vec<Vec<N>>,
}

impl<N: NeuralNode> MLP<N>
{
    fn new(layers: Vec<Vec<N>>) -> MLP<N>
    {
        MLP { layers: layers }
    }

    fn feedforward(&mut self, input: Vec<f32>) -> Vec<f32>
    {
        //TODO: There's a Rust-ier way
        let mut accum = input.clone();
        for l in &mut self.layers
        {
            let mut new_accum = vec![]; 
            for i in 0..l.len()
            {
                new_accum.push(l[i].feedforward(&accum));
            }
            accum = new_accum;
        }
        accum
    }
}


pub struct SimpleNeuralNode
{
    bias: f32,
    weights: Vec<f32>,
}

impl SimpleNeuralNode
{
    fn new(bias: f32) -> SimpleNeuralNode
    {
        SimpleNeuralNode { bias: bias, weights: vec![] }    
    }

    fn new_from_weights(bias: f32, starting_weights: Vec<f32>) -> SimpleNeuralNode
    {
        SimpleNeuralNode { bias: bias, weights: starting_weights }
    }
}

impl NeuralNode for SimpleNeuralNode
{
    fn feedforward(&mut self, inputs: &Vec<f32>) -> f32
    {
        //TODO: There's a Rust-ier way
        let mut v = self.bias;
        for i in 0..inputs.len()
        {
            v += inputs[i] * self.weights[i];
        }
        v
    }
}



#[cfg(test)]
mod test
{
    use super::*;

    #[test]
    fn test_smoke()
    {
        let snn = SimpleNeuralNode::new(0.0);
        let mlp = MLP::new(vec![vec![snn]]);
    }
}
