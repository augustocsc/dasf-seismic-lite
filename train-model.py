'''
construir um pipeline DASF para:
    extrair o atributo sísmico de interesse (--attribute) a partir do dado sísmico de entrada
    extrair features do dado sísmico
    treinar um modelo de regressão baseado em Boosted Trees 
    gravar o modelo treinado no arquivo de saída (--output).
'''
import argparse
from pathlib import Path
from typing import Callable, Tuple
import dask.array as da
import dask.dataframe as dd
import pandas as pd
import numpy as np
import time

#dask performance report
import matplotlib.pyplot as plt

from dasf_seismic.attributes.complex_trace import Envelope, InstantaneousPhase, InstantaneousFrequency
from dasf.ml.xgboost.xgboost import XGBRegressor
from dask.distributed import Client, performance_report
from dasf.transforms import ArraysToDataFrame, PersistDaskData
from dasf.pipeline import Pipeline
from dasf.datasets import Dataset
from dasf.pipeline.executors import DaskPipelineExecutor
from dasf.utils.decorators import task_handler
from dasf.transforms import Transform


class MyDataset(Dataset):
    """Classe para carregar dados de um arquivo .day
    """
    def __init__(self, name: str, data_path: str, chunks: str = "10Mb"):
        """Instancia um objeto da classe MyDataset

        Parameters
        ----------
        name : str
            Nome simbolicamente associado ao dataset
        data_path : str
            Caminho para o arquivo .zarr
        chunks: str
            Tamanho dos chunks para o dask.array
        """
        super().__init__(name=name)
        self.data_path = data_path
        self.chunks = {0: "auto", 1: -1, 2: -1}
        
    def _lazy_load_cpu(self):
        data = da.from_zarr(self.data_path, chunks=self.chunks)[:, :, :]
    
        return data
    
    def _load_cpu(self):
        return np.load(self.data_path)
    
    @task_handler
    def load(self):
        ...

class CheckType(Transform):
    def _lazy_transform_cpu(self, X):
        print((X.dtype))
        return X
    def _transform_cpu(self, X=None, **kwargs):
        print((X.dtype))
        return X

class Print(Transform):
    def _lazy_transform_cpu(self, X):
        print(X)
        return X
    def _transform_cpu(self, X=None, **kwargs):
        print(X)
        return X

class GetFeature(Transform):

    def _lazy_transform_cpu(self, X):
        
        da_y = X.iloc[:, :-1]
        print(da_y.shape)
        return da_y
    def _transform_cpu(self, X=None, **kwargs):
        return da.array(X.iloc[:, :-1].to_numpy()).persist()

class GetAttribute(Transform):
    def _lazy_transform_cpu(self, X):
        da_x = X.iloc[:, -1]
        print(da_x.shape)
        return da_x
    def _transform_cpu(self, X=None, **kwargs):
        return da.array(X.iloc[:, -1].to_numpy()).persist()

class ListToDataframe(Transform):
    
    def _lazy_transform_cpu(self, X=None, **kwargs):
        
        attribute = [kwargs["attribute"].flatten()]
        
        data = [kwargs["features_point"]]
        if kwargs["features_sample"] is not None:
            data = da.append(data, kwargs["features_sample"], axis=0)
        if kwargs["features_trace"] is not None:
            data = da.append(data, kwargs["features_trace"], axis=0)
        if kwargs["features_inline"] is not None:
            data = da.append(data, kwargs["features_inline"], axis=0)

        data = da.append(data, attribute, axis=0)

        df = pd.DataFrame(data.T)
        df = dd.from_pandas(df, npartitions=10)

        return df
    def _transform_cpu(self, X=None, **kwargs):
        #features_point = data, kwargs["features_point"]
        #features_trace = data, kwargs["features_trace"]
        #features_sample = data, kwargs["features_sample"]
        #features_inline = data, kwargs["features_inline"]
        
        #data = da.concatenate([features_point, features_trace, features_sample, features_inline], axis=0)
        #df = pd.DataFrame([features_point, features_trace, features_sample, features_inline])
        #return df
        print("CEPEUUUUUUUUUUUU")
        return None
    
class FeatureExtractor(Transform):
    """Classe para extrair atributos de um dado
    """
    def __init__(self, size=0, shift=1, axis=0, signal=1):
        """Instancia um objeto da classe FeatureExtractor

        Parameters
        ----------
        size : int, optional
            Tamanho da janela de atributos, by default 0
        shift : int, optional
            Deslocamento da janela de atributos, by default 1
        axis : int, optional
            Eixo ao longo do qual o deslocamento será aplicado, by default 0
        """
        self.size = size
        self.shift = shift
        self.axis = axis
        self.signal = signal

    def _lazy_transform_cpu(self, X):
        
        data = None
        if self.axis is None:
            data = X.flatten()
        else:
            for i in reversed(range(self.size)):
                row = da.roll(X, shift = self.signal * (i + self.shift), axis = self.axis).flatten()
                
                if data is None or data.size == 0:                  #First iteration
                    data = row
                else:
                    data = da.vstack((data, row))
                
            for i in range(self.size):
                row = da.roll(X, shift = self.signal * -(i+self.shift), axis = self.axis).flatten()            
                data = da.vstack((data, row))
        return data

    def _transform_cpu(self, X):
        print("CEPEUUUUUUUUUUUU")
        data = None
        if self.axis is None:
            data = X.flatten()
        else:
            for i in reversed(range(self.size)):
                row = np.roll(X, shift = self.signal * (i + self.shift), axis = self.axis).flatten()
                
                if data is None or data.size == 0:                  #First iteration
                    data = row
                else:
                    data = np.vstack((data, row))
                
            for i in range(self.size):
                row = np.roll(X, shift = self.signal * -(i+self.shift), axis = self.axis).flatten()            
                data = np.vstack((data, row))
        return data

            
def create_executor(address: str=None) -> DaskPipelineExecutor:
    """Cria um DASK executor

    Parameters
    ----------
    address : str, optional
        Endereço do Scheduler, by default None

    Returns
    -------
    DaskPipelineExecutor
        Um executor Dask
    """
    if address is not None:
        addr = ":".join(address.split(":")[:2])
        port = str(address.split(":")[-1])
        print(f"Criando executor. Endereço: {addr}, porta: {port}")
        return DaskPipelineExecutor(local=False, use_gpu=False, address=addr, port=port)
    else:
        return DaskPipelineExecutor(local=True, use_gpu=False)
        
def create_pipeline(dataset_path: str, executor: DaskPipelineExecutor, pipeline_save_location: str = None, attribute = 'ENVELOPE', inline = 0, trace = 0, sample = 0) -> Tuple[Pipeline, Callable]:
    """Cria o pipeline DASF para ser executado

    Parameters
    ----------
    dataset_path : str
        Caminho para o arquivo .zarr
    executor : DaskPipelineExecutor
        Executor Dask
    pipeline_save_location : str, optional
        Caminho para salvar o pipeline, by default None

    Returns
    -------
    Tuple[Pipeline, Callable]
        Uma tupla, onde o primeiro elemento é o pipeline e o segundo é último operador (kmeans.fit_predict), 
        de onde os resultados serão obtidos.
    """

    if attribute == 'ENVELOPE':
        attr    = Envelope()
    elif attribute == 'COS-INST-PHASE':
        atrr    = InstantaneousPhase()
    elif attribute == 'INST-FREQ':
        attr    = InstantaneousFrequency()
        
    print("Criando pipeline....")
    # Declarando os operadores necessários
    dataset         = MyDataset(name="F3 dataset", data_path=dataset_path)
    
    features_point  = FeatureExtractor(size=0, shift=0, axis=None)
    features_sample = FeatureExtractor(size=inline, shift=1, axis=1)
    features_trace  = FeatureExtractor(size=sample, shift=1, axis=2)
    features_inline = FeatureExtractor(size=trace, shift=1, axis=0, signal=-1)
    
    list2df         = ListToDataframe()

    get_attribute   = GetAttribute()
    get_feature     = GetFeature()

    persist_attr    = PersistDaskData()
    persist_feat    = PersistDaskData()
    show            = Print()

    model = XGBRegressor()
    
    # Compondo o pipeline
    pipeline = Pipeline(
        name="F3 seismic attributes",
        executor=executor
    )
    
    pipeline.add(dataset)

    pipeline.add(features_point,  X=dataset)
    pipeline.add(features_inline, X=dataset)
    pipeline.add(features_trace, X=dataset)
    pipeline.add(features_sample, X=dataset)
    pipeline.add(attr,      X=dataset)
    pipeline.add(list2df, features_point=features_point,features_inline=features_inline, features_trace=features_trace, features_sample=features_sample, attribute=attr)

    pipeline.add(get_attribute, X=list2df)
    pipeline.add(get_feature, X=list2df)
    

    pipeline.add(model.fit, X=get_feature, y=get_attribute)
    

    #if pipeline_save_location is not None:
#    	pipeline.visualize(filename=pipeline_save_location)
    
    return pipeline, model.fit

def run(pipeline: Pipeline, last_node: Callable) -> np.ndarray:
    """
    Executa o pipeline e retorna o resultado

    Parameters
    ----------
    pipeline : Pipeline
        Pipeline a ser executado
    last_node : Callable
        Último operador do pipeline, de onde os resultados serão obtidos

    Returns
    -------
    da.ndarray
        NumPy array com os resultados
    """
    print("Executando pipeline")
    start = time.time()
    pipeline.run()
    res = pipeline.get_result_from(last_node)
    end = time.time()
    
    print(f"Feito! Tempo de execução: {end - start:.2f} s")
    return res, end - start
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Executa o pipeline")
    parser.add_argument("--address", type=str, default=None, help="Endereço do scheduler. Formato: tcp://<ip>:<port>")
    parser.add_argument("--attribute", type=str, default=None, help="Nome do atributo a ser usado para treinar o modelo. Os valores possíveis são:\
                                                                            ENVELOPE: Envelope\
                                                                            INST-FREQ: Frequência instantânea\
                                                                            COS-INST-PHASE: Cosseno instantâneo da fase")
    parser.add_argument("--data", type=str, required=True, help="Caminho para o arquivo .zarr")

    parser.add_argument("--inline-window", type=int, default=0, help="número de vizinhos na dimensão das inlines.")
    parser.add_argument("--samples-window", type=int, default=0, help="número de vizinhos na dimensão das amostras de um traço.")
    parser.add_argument("--trace-window", type=int, default=0, help="número de vizinhos na dimensão dos traços de uma inline.")
    parser.add_argument('--output', type=str, default=None, help='Output file name')
    parser.add_argument("--save-pipeline-fig", type=str, default="pipeline-fig.png", help="Local para salvar a figura do pipeline")
    parser.add_argument("--iterations", type=int, default=1, help="Número de iterações")

    args = parser.parse_args()

    if args.output is None:
        file_name = f"run_{args.inline_window}_{args.trace_window}_{args.samples_window}"
    client = Client(args.address.replace("tcp://", ""))

    with performance_report(filename=f"data/report/{file_name}.html"):
        # Criamos o executor
        executor = create_executor(args.address)
        tempo = []
        for i in range(args.iterations):
            print(f"ITERATION {i}")
            # Depois o pipeline
            pipeline, last_node = create_pipeline(args.data, executor, pipeline_save_location=args.save_pipeline_fig, inline=args.inline_window, trace = args.trace_window, sample = args.samples_window)
            # Executamos e pegamos o resultado
            res, t = run(pipeline, last_node)
            tempo.append(t)
            res.save_model(f"data/model/{file_name}_{i}.json")
    

    #save tempo in a csv file
    import csv
    with open(f'data/time/{file_name}.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(tempo)
        
    
#00:10:23,066