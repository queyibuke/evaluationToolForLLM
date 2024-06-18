#include<iostream>
#include<string>
#include<cstdint>
#include<cmath>
#include<iomanip>
#include<vector>

std::vector<std::string> option{"1 estimate memory usage", "2 estimate traintime", "3 training parameter optimization"};
int opt = 0;

int seqLen = 0;//序列长度
std::string modelSize = {};//模型参数
uint64_t modelSizeInt = 0;//模型参数
int attnHeadNum = 0;//注意力头数
float globalBatchSize = 0;//全局批量大小
float miniBatchSize = 0;//微批量大小
int step = 0;//在一次梯度更新中累积的传播步数
int gpuNumber = 0;//GPU数量
int hiddenLayerDimension = 0;//隐藏层维度
int layer = 0;//层数

int optimizerStrategy = 0;//优化器策略 0:None 1:zero1 2:zero2 3:zero3
int dataParaSize = 0;//数据并行大小
int tensorParaSize = 0;//张量并行大小
int pipelineParaSize = 0;//流水线并行大小
int sequenceParaSize = 0;//序列并行大小

std::string dataNum = {};
float dataNumFloat = 0;
float fPointOp = 0.0;
std::string MFU = {};
float MFUfloat = 0.0;

uint64_t memUsage = 0;
uint64_t optimizerMem = 0;
uint64_t activationMem = 0;
float trainTime = 0.0;

/**
 * @brief 输入参数
 *
 * 从标准输入读取一系列参数，包括序列长度、模型大小、注意力头数、全局批处理大小、微批处理大小、步长、GPU数量、隐藏层维度、层数、优化器策略、数据并行大小、张量并行大小、管道并行大小和序列并行大小。
 *
 * @param seqLen 序列长度引用
 * @param modelSize 模型大小引用
 * @param attnHeadNum 注意力头数引用
 * @param globalBatchSize 全局批处理大小引用
 * @param miniBatchSize 微批处理大小引用
 * @param step 步长引用
 * @param gpuNumber GPU数量引用
 * @param hiddenLayerDimension 隐藏层维度引用
 * @param layer 层数引用
 * @param optimizerStrategy 优化器策略引用
 * @param dataParaSize 数据并行大小引用
 * @param tensorParaSize 张量并行大小引用
 * @param pipelineParaSize 管道并行大小引用
 * @param sequenceParaSize 序列并行大小引用
 */
void inputParaForEvalMem(int& seqLen, std::string& modelSize , int& attnHeadNum, float& globalBatchSize, float& miniBatchSize, int& step, int& gpuNumber, int& hiddenLayerDimension, int& layer,
    int& optimizerStrategy, int& dataParaSize, int& tensorParaSize, int& pipelineParaSize, int& sequenceParaSize) {
    std::cout << "please input the following parameters(delimited by Spaces):" << std::endl 
        << "sequenceLenth modelSize attentionHead globalBatchSize step gpuNumber hiddenLayerDimension layer(like 2048 1.3B 12 256 2 16 768 12):" << std::endl;
    std::cin >> seqLen >> modelSize >> attnHeadNum >> globalBatchSize >> step >> gpuNumber >> hiddenLayerDimension >> layer;
    std::cout << "optimizerStrategy dataParaSize tensorParaSize pipelineParaSize sequenceParaSize (if don't use zero 1 2 3, please input 0 for optimizerStrategy)" << std::endl 
        << "like 1 1 1 1 1:" << std::endl;
    std::cin >> optimizerStrategy >> dataParaSize >> tensorParaSize >> pipelineParaSize >> sequenceParaSize;
    while (sequenceParaSize != 0 && (optimizerStrategy == 2 || optimizerStrategy == 3)) {
        std::cout << "optimizerStrategy is zero2 or zero3 and sequenceParaSize is 0 have to choose one or the orther!" << std::endl
            << "please confirm para ande reinput optimizerStrategy and sequenceParaSize" << std::endl;
        std::cin >> optimizerStrategy >> sequenceParaSize;
    }

    miniBatchSize = globalBatchSize / (step * gpuNumber);

    std::cout << "sequenceLenth is " << seqLen << std::endl 
        << "modelSize is " << modelSize << std::endl
        << "attentionHead is " << attnHeadNum << std::endl
        << "globalBatchSize is " << globalBatchSize << std::endl
        << "miniBatchSize is " << miniBatchSize << std::endl
        << "step is " << step << std::endl
        << "gpuNumber is " << gpuNumber << std::endl
        << "hiddenLayerDimension is " << hiddenLayerDimension << std::endl
        << "layer is " << layer << std::endl;
    if(optimizerStrategy == 0) {
        std::cout << "optimizerStrategy is None" << std::endl;
    } else {
        std::cout << "optimizerStrategy is Zero" << optimizerStrategy << std::endl;
    }
    std::cout << "dataParaSize is " << dataParaSize << std::endl
        << "tensorParaSize is " << tensorParaSize << std::endl
        << "pipelineParaSize is " << pipelineParaSize << std::endl
        << "sequenceParaSize is " << sequenceParaSize << std::endl;
    }

void inputParaForEvalTime(std::string &dataNum, std::string &modelSize, int &gpuNumber, float &fPointOp, std::string &MFU) {
    std::cout << "please input the following parameters(delimited by Spaces):" << std::endl
        << "dataNum modelSize gpuNumber fPointOp(TFLOPs) MFU(like 10T 1.3B 256 1000 0.5):" << std::endl;
    std::cin >> dataNum >> modelSize >> gpuNumber >> fPointOp >> MFU;
    std::cout << "dataNum is " << dataNum << std::endl
        << "modelSize is " << modelSize << std::endl
        << "gpuNumber is " << gpuNumber << std::endl
        << "fPointOp is " << fPointOp << std::endl
        << "MFU is " << MFU << std::endl;
}

void init() {
    switch(opt) {
        case 1:
            inputParaForEvalMem(seqLen, modelSize, attnHeadNum, globalBatchSize, miniBatchSize, step, gpuNumber, hiddenLayerDimension, layer,
                optimizerStrategy, dataParaSize, tensorParaSize, pipelineParaSize, sequenceParaSize);
            break;
        case 2:
            inputParaForEvalTime(dataNum, modelSize, gpuNumber, fPointOp, MFU);
            break;
        case 3:
            //todo
            break;
        default:
            break;
    }
}

void confirmPara() {
    std:: cout << "please confirm the para, if the para is wrong, please input 'n', else input '[Y]/y':";
    char confirm;
    std::cin >> confirm;
    while(confirm != 'Y' && confirm != 'y' && confirm != 'N' && confirm != 'n') {
        std::cout << "the confirm para is wrong, please reinput the confirm para:" << std::endl;
        std::cin >> confirm;
    }
    if(confirm == 'N' || confirm == 'n') {
        std::cout << "please reinput the para:" << std::endl;
        init();
        confirmPara();
    }
}
/**
 * @brief 计算优化器所需的内存大小
 *
 * 根据优化策略计算参数、梯度和优化器状态占用显存
 *
 * @param f_optimizerStrategy 优化器策略
 * @param f_modelSize 模型大小
 * @param f_gpuNmuber GPU 数量
 *
 * @return 优化器所需的内存大小
 */
uint64_t getOptimizerMem(const int f_optimizerStrategy, const uint64_t f_modelSize, const int f_gpuNumber) {
    uint64_t optimizerMem = 0;
    switch (f_optimizerStrategy) {
    case 0:
        optimizerMem = 16 * f_modelSize;
        break;
    case 1:
        optimizerMem = 4 * f_modelSize * f_gpuNumber + 12 * f_modelSize;
        break;
    case 2:
        optimizerMem = 2 * f_modelSize * f_gpuNumber + 14 * f_modelSize;
        break;
    case 3:
        optimizerMem = 16 * f_modelSize;
        break;
    default:
        optimizerMem = 16 * f_modelSize;
        break;
    }
    return optimizerMem;
}

/**
 * @brief 计算激活内存大小
 *
 * 根据给定的参数计算激活内存的大小。
 *
 * @param f_seqLen 序列长度
 * @param f_attnHeadNum 注意力头数
 * @param f_globalBatchSize 全局批处理大小
 * @param f_minibatchSize 迷你批处理大小
 * @param f_step 一次梯度更新中累积的前向传播和反向传播的步数
 * @param f_gpuNumber GPU数量
 * @param f_hiddenLayerDimension 隐藏层维度
 * @param f_layer 层数
 * @param f_dataParaSize 数据并行大小
 * @param f_tensorParaSize 张量并行大小
 * @param f_pipelineParaSize 流水线并行大小
 * @param f_sequenceParaSize 序列并行大小
 *
 * @return 激活内存大小
 */
uint64_t getActivationMem(const int f_seqLen, const int f_attnHeadNum, const int f_globalBatchSize, const int f_minibatchSize,
    const int f_step, const int f_gpuNumber, const int f_hiddenLayerDimension, const int f_layer,
    const int f_dataParaSize, const int f_tensorParaSize, const int f_pipelineParaSize, const int f_sequenceParaSize) {
    
    uint64_t activationMem = 0;
    uint64_t a1 = 0;
    uint64_t a2 = 0;
    int t_seqLens = f_seqLen;
    int t_globalBatchSize = f_globalBatchSize;
    int t_dataParaSize = f_dataParaSize;
    int t_tensorParaSize = f_tensorParaSize;
    int t_pipelineParaSize = f_pipelineParaSize;
    int t_sequenceParaSize = f_sequenceParaSize;

    if(t_dataParaSize <= 1 ) {
        t_dataParaSize = 1;
    } else{
        t_globalBatchSize = t_globalBatchSize / f_step / f_gpuNumber;
    }
    if(t_tensorParaSize <= 1) {
        t_tensorParaSize = 1;
    }
    if(t_pipelineParaSize <= 1) {
        t_pipelineParaSize = 1;
    } else {
        t_globalBatchSize = t_globalBatchSize / t_pipelineParaSize;
    }
    if(t_sequenceParaSize <= 1) {
        t_sequenceParaSize = 1;
    } else {
        t_seqLens = t_seqLens / t_sequenceParaSize;
    }
    a1 = f_layer * t_globalBatchSize * t_seqLens;
    if (t_tensorParaSize == 1) {
        a2 = 34 * f_hiddenLayerDimension  + 5 * t_seqLens * f_attnHeadNum;
        activationMem = a1 * a2 / t_pipelineParaSize;
    } else {
        a2 = (10 + 24 / t_tensorParaSize) * f_hiddenLayerDimension  + 5 * t_seqLens * f_attnHeadNum / t_tensorParaSize;
        activationMem = a1 * a2 / t_pipelineParaSize;
    }
    return activationMem;
}

float getTrainingTime(const float f_dataNum, const uint64_t f_modelSize, const int f_gpuNumber, const float f_fPointOp, const float f_MFU) {
    
    float trainingDay = 0;
    trainingDay = f_dataNum * 6 * f_modelSize / (f_gpuNumber * f_fPointOp * f_MFU ) / 3600 / 24;
    return trainingDay;
}

int main(char **argv, int argc) {
    system("chcp 936");
    std::cout << "follow these tips to complete the model evaluation" << std::endl;
    for(auto& a : option ) {
        std::cout << a << std::endl;
    }
    std::cout << "please enter the function number: ";
    std::cin >> opt;
    while(opt <= 0 || opt > option.size()) {
        std::cout << "the number is not in the range, the input is wrong, please re-enter: " ;
        std::cin >> opt;
    }
    init();
    confirmPara();
    switch (opt) {
    case 1:
        modelSizeInt = static_cast<uint64_t>(static_cast<int>(std::stof(modelSize.substr(0, modelSize.size() - 1)) * 1000)) * 1000000;
        std::cout << "modelSize is " << modelSizeInt << '(' << modelSize << ')' << std::endl;
        optimizerMem = getOptimizerMem(optimizerStrategy, modelSizeInt, gpuNumber);
        activationMem = getActivationMem(seqLen, attnHeadNum, globalBatchSize, miniBatchSize, step, gpuNumber, hiddenLayerDimension, layer, dataParaSize, tensorParaSize, pipelineParaSize, sequenceParaSize);
        memUsage = optimizerMem + activationMem;
        std::cout << "memUsage is " << memUsage << "B(" << memUsage / 1000000000 << "GB) " << std::endl;
        break;
    case 2:
        dataNumFloat = std::stof(dataNum.substr(0, dataNum.size() - 1));
        modelSizeInt = static_cast<uint64_t>(static_cast<int>(std::stof(modelSize.substr(0, modelSize.size() - 1)) * 1000)) * 1000000;
        std::cout << "modelSize is " << modelSizeInt << '(' << modelSize << ')' << std::endl;
        if(MFU[MFU.size() - 1] == '%') {
            MFUfloat = std::stof(MFU.substr(0, MFU.size() - 1)) / 100;
        } else {
            MFUfloat = std::stof(MFU);
        }
        trainTime = getTrainingTime(dataNumFloat, modelSizeInt, gpuNumber, fPointOp, MFUfloat);
        std::cout << "trainTime is " << std::fixed << std::setprecision(1) << trainTime  << std::endl;
        break;
    case 3:
        /* code */
        break;
    default:
        break;
    }
    
    std::cout << "if you want to exit, please input 'q'. or close the win!" << std::endl;
    char c = getchar();
    while (c = getchar() != 'q');
    system("pause");
    return 0;    
}