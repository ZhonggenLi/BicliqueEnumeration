// test_CUDA_1.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//
#include<iostream>
#include<vector>
#include<map>
#include<algorithm>
#include<chrono>
#include<time.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define N 10
//#define MAXSIZE 10000 //max size of intersection size
#define MAXSTACKSIZE 10 //max stack size
//#define MAXVERTEXSIZE 31000
#define MAXHSIZE 31000
#define MAXSSIZE 10000

#define BLOCKNUM 64
#define THREADNUM 32

#define checkCudaErrors(val) check( (val), #val, __FILE__, __LINE__)

template<typename T>
void check(T err, const char* const func, const char* const file, const int line) {
	if (err != cudaSuccess) {
		std::cerr << "CUDA error at: " << file << ":" << line << std::endl;
		std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
		exit(1);
	}
}

class Vertex {
public:
	unsigned label;
	std::vector<unsigned> neighbor;
	Vertex() {}
	Vertex(unsigned lb) {
		label = lb;
	}
};

class CSR {
public:
	unsigned* row_offset = nullptr;
	unsigned* column_index = nullptr;
	CSR() {
		row_offset = nullptr;
		column_index = nullptr;
	}
	~CSR() {
		delete[] row_offset;
		delete[] column_index;
	}
};

class Graph {
public:
	std::vector<Vertex> vertices;
	unsigned vertex_num;
	unsigned edge_num;
	CSR* csr;

	Graph() {
		vertex_num = 0;
		edge_num = 0;
		csr = NULL;
	}

	void addVertex(unsigned lb);
	void addBipartiteEdge(unsigned lb1, unsigned lb2);
	void addEdge(unsigned lb1, unsigned lb2);
	void printGraph();
	void transformToCSR(CSR& _csr);
};

void
Graph::addVertex(unsigned lb) {
	vertices.push_back(Vertex(lb));
	vertex_num++;
}

void
Graph::addEdge(unsigned lb1, unsigned lb2) {
	vertices[lb1 - 1].neighbor.push_back(lb2);
	vertices[lb2 - 1].neighbor.push_back(lb1);
	edge_num++;
}

void
Graph::addBipartiteEdge(unsigned lb1, unsigned lb2) {
	vertices[lb1 - 1].neighbor.push_back(lb2);
	edge_num++;
}

void
Graph::printGraph() {
	std::cout << "Number of vertices: " << this->vertex_num << std::endl;
}

void
Graph::transformToCSR(CSR& _csr) {
	unsigned offset_size = this->vertex_num + 1;
	_csr.row_offset = new unsigned[offset_size];
	_csr.row_offset[0] = 0;
	unsigned sum = 0;
	for (int i = 1; i < offset_size; i++) {
		sum += this->vertices[i - 1].neighbor.size();
		_csr.row_offset[i] = sum;
	}
	//sum += this->vertices[offset_size - 1].neighbor.size();

	_csr.column_index = new unsigned[sum];
	unsigned k = 0;
	for (int i = 0; i < offset_size-1; i++) {
		for (int j = 0; j < this->vertices[i].neighbor.size(); j++) {
			_csr.column_index[k] = this->vertices[i].neighbor[j];
			k++;
		}
	}
	this->csr = &_csr;
}

void readFile(Graph& graph, bool flag) {
	FILE* fp = NULL;
	fopen_s(&fp, "youtube.txt", "r");
	int L_vertex_num, R_vertex_num, edge_num;
	fscanf_s(fp, "%d %d %d\n", &L_vertex_num, &R_vertex_num, &edge_num);
	// if flag is true, input the left layer
	if (flag == true) {
		for (unsigned i = 1; i <= L_vertex_num; i++) {
			graph.addVertex(i);
		}
		for (int j = 0; j < edge_num; j++) {
			unsigned in, out;
			fscanf_s(fp, "%u\t%u\n", &in, &out);
			graph.addBipartiteEdge(in, out);
		}
	}
	else {
		for (unsigned i = 1; i <= R_vertex_num; i++) {
			graph.addVertex(i);
		}
		for (int j = 0; j < edge_num; j++) {
			unsigned in, out;
			fscanf_s(fp, "%u\t%u\n", &in, &out);
			graph.addBipartiteEdge(out, in);
		}
	}
	fclose(fp);
}

void Collect2Hop(Graph& L, Graph& R, Graph& H, int q) {
	//for L
	unsigned L_num = L.vertex_num;
	for (unsigned m = 1; m <= L_num; m++) {
		H.addVertex(m);
	}
	for (unsigned i = 0; i < L_num; i++) {
		unsigned* list2hop = new unsigned[L_num]();
		int i_neighbor_num = L.vertices[i].neighbor.size();
		for (int j = 0; j < i_neighbor_num; j++) {
			unsigned j_vertex = L.vertices[i].neighbor[j];
			int j_neighbor_num = R.vertices[j_vertex - 1].neighbor.size();
			for (int k = 0; k < j_neighbor_num; k++) {
				list2hop[R.vertices[j_vertex - 1].neighbor[k] - 1]++;
			}
		}
		for (unsigned l = i + 1; l < L_num; l++) {
			if (list2hop[l] >= q) {
				H.addEdge(i + 1, l + 1);
			}
		}
		delete[] list2hop;
	}
}

bool cmp(std::pair<unsigned, unsigned> a, std::pair<unsigned, unsigned> b) {
	return a.second > b.second;
}

void edgeDirectingByDegree(Graph& H) {
	std::vector<std::pair<unsigned, unsigned>> lb_degree;
	unsigned vertex_num = H.vertex_num;
	//int count = 0;
	for (unsigned i = 0; i < vertex_num; i++) {
		lb_degree.push_back(std::pair<unsigned, unsigned>(i + 1, H.vertices[i].neighbor.size()));
	}
	sort(lb_degree.begin(), lb_degree.end(), cmp);
	/*for (int j = vertex_num - 1; j >= 0; j--) {
		for (int k = j - 1; k >= 0; k--) {
			std::vector<unsigned>::iterator find_val = find(H.vertices[lb_degree[j].first - 1].neighbor.begin(), H.vertices[lb_degree[j].first - 1].neighbor.end(), lb_degree[k].first);
			if (find_val != H.vertices[lb_degree[j].first - 1].neighbor.end()) {
				H.vertices[lb_degree[j].first - 1].neighbor.erase(find_val);
				count1++;
			}
		}
	}
	std::cout << "Deleted:" << count1 << std::endl;*/
	for (int j = 0; j < vertex_num; j++) {
		for (auto val : H.vertices[lb_degree[j].first - 1].neighbor) {
			std::vector<unsigned>::iterator find_val = find(H.vertices[val - 1].neighbor.begin(), H.vertices[val - 1].neighbor.end(), lb_degree[j].first);
			if (find_val != H.vertices[val - 1].neighbor.end()) {
				H.vertices[val - 1].neighbor.erase(find_val);
				//count++;
			}
		}
	}
	//std::cout << "Deleted:" << count << std::endl;
}

unsigned OrderMul(int m, int n) {
	if (n == 0 || n == m) {
		return 1;
	}
	return OrderMul(m - 1, n) + OrderMul(m - 1, n - 1);
}

void InterSection(std::vector<unsigned>& A, std::vector<unsigned>& B, std::vector<unsigned>& res) {
	// size of B is smaller than size of A
	res.clear();
	for (auto val : B) {
		auto first = std::find(A.begin(), A.end(), val);
		if (!(first == A.end())) {
			res.push_back(val);
		}
	}
}

__global__ void add(int* a, int* b, int* c) {
	int tid = blockIdx.x;
	if (tid < N) {
		c[tid] = a[tid] + b[tid];
	}
}

//row_dev, col_dev, vertex - 1, &S[level - 1][0], S_tmp, L.vertices[vertex - 1].neighbor.size(), S[level - 1].size()
__global__ void Intersect1(unsigned* row_dev, unsigned* col_dev, unsigned node, unsigned* B, unsigned* Res, int Asize, int Bsize) {
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	while (tid < Bsize) {
		unsigned val = B[tid];
		//binary search
		int ret = 0, tmp = Asize;
		while (tmp > 1) {
			int halfsize = tmp / 2;
			int cand = col_dev[row_dev[node] + ret + halfsize];
			ret += (cand < val) ? halfsize : 0;
			tmp -= halfsize;
		}
		ret += (col_dev[row_dev[node] + ret] < val);
		Res[tid] = (ret <= (Asize - 1) ? (col_dev[row_dev[node] + ret] == val) ? val : 0 : 0);
		tid += blockDim.x * gridDim.x;
	}
}

__global__ void Intersect2(unsigned* row_dev, unsigned* col_dev, unsigned node, unsigned* B, unsigned* Res, int Asize, int Bsize) {
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	while (tid < Asize) {
		unsigned val = col_dev[row_dev[node] + tid];
		//binary search
		int ret = 0, tmp = Bsize;
		while (tmp > 1) {
			int halfsize = tmp / 2;
			//printf("%d ", tid);
			//printf("%u ", B[ret + halfsize]);
			int cand = B[ret + halfsize];
			ret += (cand < val) ? halfsize : 0;
			tmp -= halfsize;
		}
		ret += (B[ret] < val);
		Res[tid] = (ret <= (Bsize - 1) ? (B[ret] == val) ? val : 0 : 0);
		tid += blockDim.x * gridDim.x;
	}
}

__device__  unsigned OrderMulDev(int m, int n) {
	if (n == 0 || n == m) {
		return 1;
	}
	return OrderMulDev(m - 1, n) + OrderMulDev(m - 1, n - 1);
}

//__device__ bool visited[BLOCKNUM][MAXSTACKSIZE][MAXHSIZE] = { 0 };

__device__ unsigned stack[BLOCKNUM][MAXSTACKSIZE];
__device__ unsigned subH[BLOCKNUM][MAXSTACKSIZE][MAXHSIZE] = { 0 };
__device__ unsigned S[BLOCKNUM][MAXSTACKSIZE][MAXSSIZE] = { 0 };

__device__ unsigned GCL[BLOCKNUM] = { 0 };
__device__ unsigned BNR[BLOCKNUM] = { 0 };
__device__ unsigned BNN[BLOCKNUM][MAXSTACKSIZE] = { 0 };
__device__ int glock = 1;

//__device__ unsigned intersect_tmplist[BLOCKNUM][MAXSSIZE] = { 0 };
//__device__ unsigned subHtmp[BLOCKNUM][MAXHSIZE] = { 0 };

__device__ void IntersectionDev(unsigned vertex, unsigned* col_dev_L, int L_neighbor_size, unsigned* B, int B_size, unsigned* Res, int& res_num, int& lock) {
	int tid_t = threadIdx.x;
	//printf("%d ", tid);
	while (tid_t < B_size) {
		unsigned val = B[tid_t];
		//binary search
		int ret = 0, tmp = L_neighbor_size;
		while (tmp > 1) {
			int halfsize = tmp / 2;
			int cand = col_dev_L[vertex + ret + halfsize];
			ret += (cand < val) ? halfsize : 0;
			tmp -= halfsize;
		}
		ret += (col_dev_L[vertex + ret] < val);
		//__syncthreads();
		if (ret <= (L_neighbor_size - 1) && col_dev_L[vertex + ret] == val)
		{
			//__syncthreads();
			for (int i = 0; i < 32; i++) {
				// Check if it is this thread's turn
				if (tid_t % 32 != i)
					continue;

				// Lock
				while (atomicExch(&lock, 0) == 0);
				// Work
				res_num++;
				Res[res_num] = val;
				// Unlock
				lock = 1;
				//__syncthreads();
			}
			//Res[atomicAdd(&res_num, 1)] = val;
		}
		//__syncthreads();

		//Res[tid_t] = (ret <= (L_neighbor_size - 1) ? (col_dev_L[vertex + ret] == val) ? val : 0 : 0);
		//__syncthreads();
		tid_t += blockDim.x;
	}
}

//GPU baseline
__global__ void findCliqueGPUNew(unsigned* row_dev_L, unsigned* col_dev_L, unsigned* row_dev_H, unsigned* col_dev_H, int* count, int* p, int* q, int* Hsize, unsigned* non_vertex) {
	__shared__ int top, level, H_neighbor_size, L_neighbor_size, tid, flag, res_num, lock, subH_size, S_level_size;
	__shared__ unsigned vertex, num_tmp;
	//__shared__ double duration;
	tid = blockIdx.x;
	int B_size = 0, B_size_H = 0;
	//clock_t start, end;
	//start = clock();
	__syncthreads();
	while (tid < *Hsize) {

		//unsigned S_level_size = 0;
		if (threadIdx.x == 0) {
			//clock_t start, end;
			//start = clock();
			subH_size = 0, L_neighbor_size = 0, B_size = 0, H_neighbor_size = 0, B_size_H = 0, flag = 0, level = 0, top = -1, S_level_size = 0,num_tmp = 0;
			stack[blockIdx.x][++top] = 0;

			subH_size = row_dev_H[non_vertex[tid] + 1] - row_dev_H[non_vertex[tid]];
			subH[blockIdx.x][0][0] = subH_size;
		}
		__syncthreads();
		for (int i = threadIdx.x; i < subH_size; i += blockDim.x) {
			subH[blockIdx.x][0][i + 1] = col_dev_H[row_dev_H[non_vertex[tid]] + i];
		}
		if (threadIdx.x == 0) {
			//__syncthreads();
			//visited[blockIdx.x][tid] = 1;
			S_level_size = row_dev_L[non_vertex[tid] + 1] - row_dev_L[non_vertex[tid]];
			S[blockIdx.x][0][0] = S_level_size;
		}
		__syncthreads();
		for (int i = threadIdx.x; i < S_level_size; i += blockDim.x) {
			S[blockIdx.x][0][i + 1] = col_dev_L[row_dev_L[non_vertex[tid]] + i];
		}
		//end = clock();
		//duration += (((double)end - start) / CLOCKS_PER_SEC);
		//printf("%d, %lf\n", tid, duration);
		__syncthreads();
		//level++;
		while (top != -1) {
			__syncthreads();
			//unsigned vertex = stack[top];
			if (level == *p - 1) {
				if (threadIdx.x == 0) {
					//*count += OrderMulDev(S[blockIdx.x][level - 1][0], *q);
					atomicAdd(count, OrderMulDev(S[blockIdx.x][level][0], *q));
					//printf("%d ", *count);
					stack[blockIdx.x][top] = 0;
					top--;
					level--;
				}
			}

			if (threadIdx.x == 0) {
				flag = 0;
			}

			__syncthreads();
			//printf("%d ", subH[level][0]);
			int size = subH[blockIdx.x][level][0];
			for (int j = stack[blockIdx.x][top]; j < size; j++) {
				if (threadIdx.x == 0) {
					vertex = subH[blockIdx.x][level][j + 1] - 1;
					flag = 1;
					stack[blockIdx.x][top] = j + 1;
					top++;
					level++;
					L_neighbor_size = row_dev_L[vertex + 1] - row_dev_L[vertex];
					res_num = 0;
					lock = 1;
					//start = clock();
					num_tmp += S[blockIdx.x][level - 1][0];
				}
				//__syncthreads();
				__syncthreads();
				IntersectionDev(row_dev_L[vertex], col_dev_L, L_neighbor_size, &S[blockIdx.x][level - 1][1], S[blockIdx.x][level - 1][0], &S[blockIdx.x][level][0], res_num, lock);
				__syncthreads();
				//checkCudaErrors(cudaGetLastError());
				if (threadIdx.x == 0) {
					//end = clock();
					//duration = ((double)(end - start) / CLOCKS_PER_SEC);
					//printf("%lf\n", duration);
					//intersect_num = 0;
					//__syncthreads();
					S[blockIdx.x][level][0] = res_num;
					H_neighbor_size = row_dev_H[vertex + 1] - row_dev_H[vertex];
					//B_size_H = subH[level - 1][0];
					res_num = 0;
					lock = 1;
					num_tmp += subH[blockIdx.x][level - 1][0];
				}
				__syncthreads();
				IntersectionDev(row_dev_H[vertex], col_dev_H, H_neighbor_size, &subH[blockIdx.x][level - 1][1], subH[blockIdx.x][level - 1][0], &subH[blockIdx.x][level][0], res_num, lock);
				__syncthreads();

				//checkCudaErrors(cudaGetLastError());
				if (threadIdx.x == 0) {
					//__syncthreads();
					subH[blockIdx.x][level][0] = res_num;
					if (S[blockIdx.x][level][0] < *q || subH[blockIdx.x][level][0] < *p - level - 1) {
						stack[blockIdx.x][top] = 0;
						top--;
						level--;
					}
				}
				__syncthreads();
				break;
			}
			__syncthreads();
			if (flag == 0) {
				if (threadIdx.x == 0) {
					stack[blockIdx.x][top] = 0;
					top--;
					level = level == 0 ? 0 : level - 1;
				}
			}
			__syncthreads();
		}
		if (threadIdx.x == 0) {
			tid += gridDim.x;
			//printf("H:%u\n", num_tmp);
		}
		
	}
	__syncthreads();
	//end = clock();
	if (threadIdx.x == 0) {
		printf("%d end\n", blockIdx.x);
	}
}

__global__ void findCliqueGPUStre2(unsigned* row_dev_L, unsigned* col_dev_L, unsigned* row_dev_H, unsigned* col_dev_H, int* count, int* p, int* q, int* Hsize, unsigned* non_vertex) {
	__shared__ int top, level, H_neighbor_size, L_neighbor_size, tid, flag, res_num, lock, subH_size, S_level_size;
	__shared__ unsigned vertex, num_tmp, visited_root, visited_second, min, mask;
	//__shared__ double duration;
	tid = blockIdx.x;
	int B_size = 0, B_size_H = 0;
	if (threadIdx.x == 0) {
		visited_root = 0;
		visited_second = 0;
	}
	//clock_t start, end;
	//start = clock();
	__syncthreads();
	while (tid < *Hsize) {

		//unsigned S_level_size = 0;
		if ((visited_root != 0 && non_vertex[tid] <= visited_root) || (BNR[blockIdx.x] > 0 && tid < BNR[blockIdx.x])) {
			tid += gridDim.x;
			continue;
		}
		if (threadIdx.x == 0) {
			//clock_t start, end;
			//start = clock();
			printf("%d:%d\n", blockIdx.x, tid);
			subH_size = 0, L_neighbor_size = 0, B_size = 0, H_neighbor_size = 0, B_size_H = 0, flag = 0, level = 0, top = -1, S_level_size = 0, num_tmp = 0;
			stack[blockIdx.x][++top] = 0;

			subH_size = row_dev_H[non_vertex[tid] + 1] - row_dev_H[non_vertex[tid]];
			subH[blockIdx.x][0][0] = subH_size;
		}
		__syncthreads();
		for (int i = threadIdx.x; i < subH_size; i += blockDim.x) {
			subH[blockIdx.x][0][i + 1] = col_dev_H[row_dev_H[non_vertex[tid]] + i];
		}
		if (threadIdx.x == 0) {
			//__syncthreads();
			//visited[blockIdx.x][tid] = 1;
			S_level_size = row_dev_L[non_vertex[tid] + 1] - row_dev_L[non_vertex[tid]];
			S[blockIdx.x][0][0] = S_level_size;
		}
		__syncthreads();
		for (int i = threadIdx.x; i < S_level_size; i += blockDim.x) {
			S[blockIdx.x][0][i + 1] = col_dev_L[row_dev_L[non_vertex[tid]] + i];
		}
		//end = clock();
		//duration += (((double)end - start) / CLOCKS_PER_SEC);
		//printf("%d, %lf\n", tid, duration);
		__syncthreads();
		//level++;
		while (top != -1) {
			__syncthreads();
			//unsigned vertex = stack[top];
			if (level == *p - 1) {
				if (threadIdx.x == 0) {
					//*count += OrderMulDev(S[blockIdx.x][level - 1][0], *q);
					atomicAdd(count, OrderMulDev(S[blockIdx.x][level][0], *q));
					//printf("%d ", *count);
					stack[blockIdx.x][top] = 0;
					top--;
					level--;
					//next root has been stolen
					if (GCL[blockIdx.x] >> 30 == 1) visited_root = (GCL[blockIdx.x] << 3) >> 3;
					else if (GCL[blockIdx.x] >> 30 == 2) visited_second = (GCL[blockIdx.x] << 3) >> 3;
					GCL[blockIdx.x] = tid / gridDim.x * (*p) + level;
				}
			}

			if (threadIdx.x == 0) {
				flag = 0;
			}

			__syncthreads();
			//printf("%d ", subH[level][0]);
			int size = subH[blockIdx.x][level][0];
			for (int j = stack[blockIdx.x][top]; j < size; j++) {
				if (threadIdx.x == 0) {
					vertex = subH[blockIdx.x][level][j + 1] - 1;
				}
				if ((top == 0 && visited_second != 0 && vertex <= visited_second) || (top == 0 && BNN[blockIdx.x][0]!= 0 && vertex <= BNN[blockIdx.x][0])) {
					if (threadIdx.x == 0) {
						stack[blockIdx.x][top] = BNN[blockIdx.x][0] + 1;
						printf("%d Skip %d\n", blockIdx.x, BNN[blockIdx.x][0]);
						visited_second = 0;
					}
					continue;
				}
				if (threadIdx.x == 0) {
					flag = 1;
					stack[blockIdx.x][top] = j + 1;
					top++;
					level++;
					//next root has been stolen
					if (GCL[blockIdx.x] >> 30 == 1) visited_root = (GCL[blockIdx.x] << 3) >> 3;
					else if (GCL[blockIdx.x] >> 30 == 2) visited_second = (GCL[blockIdx.x] << 3) >> 3;
					GCL[blockIdx.x] = tid / gridDim.x * (*p) + level;
					L_neighbor_size = row_dev_L[vertex + 1] - row_dev_L[vertex];
					res_num = 0;
					lock = 1;
					//start = clock();
					num_tmp += S[blockIdx.x][level - 1][0];
				}
				//__syncthreads();
				__syncthreads();
				IntersectionDev(row_dev_L[vertex], col_dev_L, L_neighbor_size, &S[blockIdx.x][level - 1][1], S[blockIdx.x][level - 1][0], &S[blockIdx.x][level][0], res_num, lock);
				__syncthreads();
				//checkCudaErrors(cudaGetLastError());
				if (threadIdx.x == 0) {
					//end = clock();
					//duration = ((double)(end - start) / CLOCKS_PER_SEC);
					//printf("%lf\n", duration);
					//intersect_num = 0;
					//__syncthreads();
					S[blockIdx.x][level][0] = res_num;
					H_neighbor_size = row_dev_H[vertex + 1] - row_dev_H[vertex];
					//B_size_H = subH[level - 1][0];
					res_num = 0;
					lock = 1;
					num_tmp += subH[blockIdx.x][level - 1][0];
				}
				__syncthreads();
				IntersectionDev(row_dev_H[vertex], col_dev_H, H_neighbor_size, &subH[blockIdx.x][level - 1][1], subH[blockIdx.x][level - 1][0], &subH[blockIdx.x][level][0], res_num, lock);
				__syncthreads();

				//checkCudaErrors(cudaGetLastError());
				if (threadIdx.x == 0) {
					//__syncthreads();
					subH[blockIdx.x][level][0] = res_num;
					if (S[blockIdx.x][level][0] < *q || subH[blockIdx.x][level][0] < *p - level - 1) {
						stack[blockIdx.x][top] = 0;
						top--;
						level--;
						//next root has been stolen
						if (GCL[blockIdx.x] >> 30 == 1) visited_root = (GCL[blockIdx.x] << 3) >> 3;
						else if (GCL[blockIdx.x] >> 30 == 2) visited_second = (GCL[blockIdx.x] << 3) >> 3;
						GCL[blockIdx.x] = tid / gridDim.x * (*p) + level;
					}
				}
				__syncthreads();
				break;
			}
			__syncthreads();
			if (flag == 0) {
				if (threadIdx.x == 0) {
					stack[blockIdx.x][top] = 0;
					top--;
					level = level == 0 ? 0 : level - 1;
					//next root has been stolen
					if (GCL[blockIdx.x] >> 30 == 1) visited_root = (GCL[blockIdx.x] << 3) >> 3;
					else if (GCL[blockIdx.x] >> 30 == 2) visited_second = (GCL[blockIdx.x] << 3) >> 3;
					GCL[blockIdx.x] = tid / gridDim.x * (*p) + level;
				}
			}
			__syncthreads();
		}
		if (threadIdx.x == 0) {
			tid += gridDim.x;
			//printf("H:%u\n", num_tmp);
		}

	}
	__syncthreads();
	if (threadIdx.x == 0) GCL[blockIdx.x] = 0xFFFFFFFF;
	//end = clock();
	/*if (threadIdx.x == 0) {
		printf("%d:%lf\n", blockIdx.x, ((double)(end-start)));
	}*/
	if(threadIdx.x==0) printf("Begin tid:%d, blockId:%d\n", tid,blockIdx.x);
	/*if (blockIdx.x == 23) {
		printf("A");
	}*/
	unsigned next_vertex, idx; // mask denote it belongs to which label
	while (true) {
		if (threadIdx.x == 0) {
			min = 0xFFFFFFFF, idx = -1, mask = 0;
			//while (atomicExch(&glock, 0) == 0);
			for (int i = 0; i < gridDim.x; i++) {
				if (i == blockIdx.x) continue;
				if ((GCL[i] & 0x20000000) != 0x20000000 && GCL[i] >> 30 == 0) {
					unsigned tmp = (GCL[i] << 3) >> 3;
					if (min > tmp) {
						min = tmp;
						idx = i;
					}
				}
			}
			//glock = 1;
		}
		__syncthreads();
		if (min == 0xFFFFFFFF) {
			if (threadIdx.x == 0) {
				GCL[blockIdx.x] = 0xFFFFFFFF;
			}
			break;
		}
		__syncthreads();
		if(threadIdx.x == 0){
			//printf("%d: %d\n", blockIdx.x, min);
			//steal root
			unsigned current_root = (min - min % (*p)) / (*p) * gridDim.x + idx;
			if (GCL[idx] >> 30 != 1 && current_root + gridDim.x < *Hsize && BNR[idx] < *Hsize) {
				mask = 0x40000000;
				int next_idx = (current_root + gridDim.x) > BNR[idx] ? (current_root + gridDim.x) : BNR[idx];
				next_vertex = non_vertex[next_idx];
				GCL[idx] = next_vertex | 0x60000000; // the 30th bit denotes that the next root is stolen
				GCL[blockIdx.x] = 0x40000000;
				BNR[idx] = next_idx + gridDim.x;
				printf("%d Steal %d:%d\n", blockIdx.x, idx, next_idx);
				//# begin initializing
				subH_size = 0, L_neighbor_size = 0, B_size = 0, H_neighbor_size = 0, B_size_H = 0, flag = 0, level = 0, top = -1, S_level_size = 0, num_tmp = 0;
				stack[blockIdx.x][++top] = 0;

				subH_size = row_dev_H[next_vertex + 1] - row_dev_H[next_vertex];
				subH[blockIdx.x][0][0] = subH_size;
				for (int i = 0; i < subH_size; i++) {
					subH[blockIdx.x][0][i + 1] = col_dev_H[row_dev_H[next_vertex] + i];
				}
				S_level_size = row_dev_L[next_vertex + 1] - row_dev_L[next_vertex];
				S[blockIdx.x][0][0] = S_level_size;
				for (int i = 0; i < S_level_size; i++) {
					S[blockIdx.x][0][i + 1] = col_dev_L[row_dev_L[next_vertex] + i];
				}
				//#end initializing
			}

			//steal second level node
			/*else if (row_dev_H[current_root + 1] - row_dev_H[current_root] > stack[idx][1] && row_dev_H[current_root + 1] - row_dev_H[current_root] > BNN[idx][0] + 1) {
				mask = 0x80000000;
				int next_id = stack[idx][1] > BNN[idx][0] ? stack[idx][1] : BNN[idx][0] + 1;
				next_vertex = col_dev_H[row_dev_H[current_root] + next_id];
				GCL[idx] = next_vertex | 0xA0000000; // the 30th bit denotes the next sibling is stloen
				GCL[blockIdx.x] = 0x80000000;
				BNN[idx][0] = next_vertex;
				printf("%d steal %d's %d second: %d\n", blockIdx.x, idx, current_root, next_vertex);
				//# begin initializing
				subH_size = 0, L_neighbor_size = 0, B_size = 0, H_neighbor_size = 0, B_size_H = 0, flag = 0, level = 0, top = -1, S_level_size = 0, num_tmp = 0;
				stack[blockIdx.x][++top] = 0;

				subH_size = row_dev_H[current_root + 1] - row_dev_H[current_root];
				subH[blockIdx.x][0][0] = subH_size;
				for (int i = 0; i < subH_size; i++) {
					subH[blockIdx.x][0][i + 1] = col_dev_H[row_dev_H[current_root] + i];
				}
				S_level_size = row_dev_L[current_root + 1] - row_dev_L[current_root];
				S[blockIdx.x][0][0] = S_level_size;
				for (int i = 0; i < S_level_size; i++) {
					S[blockIdx.x][0][i + 1] = col_dev_L[row_dev_L[current_root] + i];
				}
				//#end initializing
			}*/
			//steal the lower level node except root and second level node
			/*else {
				int lev_tmp = 2, lev_cur = min % gridDim.x;
				for (lev_tmp = 2; lev_tmp < lev_cur; lev_tmp++) {
					if (stack[idx][lev_tmp] <= subH[idx][lev_tmp][0]) break;
				}
				// find a next vertex that is uncle or sibling
				if (lev_tmp < lev_cur) {
					mask = 0xC0000000;
					next_vertex = subH[idx][lev_tmp][stack[idx][lev_tmp]];
					GCL[idx] = next_vertex | 0xE0000000; // the 30th bit denotes the next sibling is stloen
					GCL[blockIdx.x] = 0xC0000000;
					printf("%d steal uncle %d:%d\n", blockIdx.x, idx, current_root);
					//# begin initializing
					subH_size = 0, L_neighbor_size = 0, B_size = 0, H_neighbor_size = 0, B_size_H = 0, flag = 0, level = lev_tmp - 1, top = lev_tmp - 1, S_level_size = 0, num_tmp = 0;
					for (int i = 0; i <= top; i++) {
						stack[blockIdx.x][i] = 0;
						subH[blockIdx.x][i][0] = 1;
					}
					// warning: S[idx][level]的元素可能会发生改变
					for (int j = 0; j <= S[idx][level][0]; j++) {
						S[blockIdx.x][level][j] = S[idx][level][j];
					}
					//#end initializing
				}
				// warning: 要不要把该GCL标记一下，下次不再遍历它
			}*/
		}
		__syncthreads();
		// process the sibling case singlely
		if (mask == 0x80000000 || mask == 0xC0000000) {
			if (threadIdx.x == 0) {
				vertex = next_vertex;
				flag = 1;
				stack[blockIdx.x][top] = 0;
				top++;
				level++;
				//GCL[blockIdx.x] = (tid / gridDim.x * (*p) + level) | mask;
				L_neighbor_size = row_dev_L[vertex + 1] - row_dev_L[vertex];
				res_num = 0;
				lock = 1;
				//start = clock();
				num_tmp += S[blockIdx.x][level - 1][0];
			}
			__syncthreads();
			IntersectionDev(row_dev_L[vertex], col_dev_L, L_neighbor_size, &S[blockIdx.x][level - 1][1], S[blockIdx.x][level - 1][0], &S[blockIdx.x][level][0], res_num, lock);
			__syncthreads();
			if (threadIdx.x == 0) {
				S[blockIdx.x][level][0] = res_num;
				H_neighbor_size = row_dev_H[vertex + 1] - row_dev_H[vertex];
				res_num = 0;
				lock = 1;
				num_tmp += subH[blockIdx.x][level - 1][0];
			}
			__syncthreads();
			IntersectionDev(row_dev_H[vertex], col_dev_H, H_neighbor_size, &subH[blockIdx.x][level - 1][1], subH[blockIdx.x][level - 1][0], &subH[blockIdx.x][level][0], res_num, lock);
			__syncthreads();
			if (threadIdx.x == 0) {
				subH[blockIdx.x][level - 1][0] = 1;
				subH[blockIdx.x][level - 1][1] = vertex;
				subH[blockIdx.x][level][0] = res_num;
				if (S[blockIdx.x][level][0] < *q || subH[blockIdx.x][level][0] < *p - level - 1) {
					top = -1;
				}
			}
			__syncthreads();
		
		}
		else if (mask == 0) {
			continue;
		}

		__syncthreads();
		while (top != -1) {
			__syncthreads();
			//unsigned vertex = stack[top];
			if (level == *p - 1) {
				if (threadIdx.x == 0) {
					//*count += OrderMulDev(S[blockIdx.x][level - 1][0], *q);
					atomicAdd(count, OrderMulDev(S[blockIdx.x][level][0], *q));
					//printf("%d ", *count);
					stack[blockIdx.x][top] = 0;
					top--;
					level--;
					GCL[blockIdx.x] = (tid / gridDim.x * (*p) + level) | mask;
				}
			}

			if (threadIdx.x == 0) {
				flag = 0;
			}

			__syncthreads();
			//printf("%d ", subH[level][0]);
			int size = subH[blockIdx.x][level][0];
			for (int j = stack[blockIdx.x][top]; j < size; j++) {
				if (threadIdx.x == 0) {
					vertex = subH[blockIdx.x][level][j + 1] - 1;
					flag = 1;
					stack[blockIdx.x][top] = j + 1;
					top++;
					level++;
					GCL[blockIdx.x] = (tid / gridDim.x * (*p) + level) | mask;
					L_neighbor_size = row_dev_L[vertex + 1] - row_dev_L[vertex];
					res_num = 0;
					lock = 1;
					//start = clock();
					num_tmp += S[blockIdx.x][level - 1][0];
				}
				//__syncthreads();
				__syncthreads();
				IntersectionDev(row_dev_L[vertex], col_dev_L, L_neighbor_size, &S[blockIdx.x][level - 1][1], S[blockIdx.x][level - 1][0], &S[blockIdx.x][level][0], res_num, lock);
				__syncthreads();
				//checkCudaErrors(cudaGetLastError());
				if (threadIdx.x == 0) {
					//end = clock();
					//duration = ((double)(end - start) / CLOCKS_PER_SEC);
					//printf("%lf\n", duration);
					//intersect_num = 0;
					//__syncthreads();
					S[blockIdx.x][level][0] = res_num;
					H_neighbor_size = row_dev_H[vertex + 1] - row_dev_H[vertex];
					//B_size_H = subH[level - 1][0];
					res_num = 0;
					lock = 1;
					num_tmp += subH[blockIdx.x][level - 1][0];
				}
				__syncthreads();
				IntersectionDev(row_dev_H[vertex], col_dev_H, H_neighbor_size, &subH[blockIdx.x][level - 1][1], subH[blockIdx.x][level - 1][0], &subH[blockIdx.x][level][0], res_num, lock);
				__syncthreads();

				//checkCudaErrors(cudaGetLastError());
				if (threadIdx.x == 0) {
					//__syncthreads();
					subH[blockIdx.x][level][0] = res_num;
					if (S[blockIdx.x][level][0] < *q || subH[blockIdx.x][level][0] < *p - level - 1) {
						stack[blockIdx.x][top] = 0;
						top--;
						level--;
						GCL[blockIdx.x] = (tid / gridDim.x * (*p) + level) | mask;
					}
				}
				__syncthreads();
				break;
			}
			__syncthreads();
			if (flag == 0) {
				if (threadIdx.x == 0) {
					stack[blockIdx.x][top] = 0;
					top--;
					level = level == 0 ? 0 : level - 1;
					GCL[blockIdx.x] = (tid / gridDim.x * (*p) + level) | mask;
				}
			}
			__syncthreads();
		}
		if (threadIdx.x == 0) {
			GCL[blockIdx.x] = 0xFFFFFFFF;
		}
		//printf("AA\n");
	}
	if (threadIdx.x == 0) printf("%d End\n", blockIdx.x);
}


//use GPU to do set intersection
void findCliqueFirstGPU(Graph& H, Graph& L, int p, int q, int level, std::vector<std::vector<unsigned>>& S, int& count, std::vector<unsigned> subH, float& STime, float& HTime, float& MTime, unsigned* row_dev, unsigned* col_dev) {

	if (level == p) {
		//std::cout << "level arrive:" << S[level - 1].size() << "," << q << std::endl;
		/*for (auto xx : ans) {
			std::cout << xx << ",";
		}
		std::cout << std::endl;
		for (auto yy : S[level-1]) {
			std::cout << yy << ",";
		}
		std::cout << std::endl;*/
		if (S[level - 1].size() >= q) {
			auto start0 = std::chrono::high_resolution_clock::now();
			count += OrderMul(S[level - 1].size(), q);
			auto end0 = std::chrono::high_resolution_clock::now();
			std::chrono::duration<float> duration0 = end0 - start0;
			MTime += duration0.count();
		}
		return;
	}
	for (auto vertex : subH) {
		if (level == 0) {
			//ans.clear();
			S.clear();

			S.push_back(L.vertices[vertex - 1].neighbor);
		}
		else {

			if (S.size() <= level) {
				S.push_back(std::vector<unsigned>());
			}
			if (L.vertices[vertex - 1].neighbor.size() > S[level - 1].size()) {

				//origin
				//res_list.push_back(std::pair<unsigned,unsigned>(L.vertices[vertex - 1].neighbor.size(), S[level - 1].size()));
				//std::cout << "first size: " << L.vertices[vertex - 1].neighbor.size() << ";second size: " << S[level - 1].size() << std::endl;
				int last_size = S[level - 1].size();
				unsigned* S_tmp = new unsigned[last_size]();

				unsigned* S_tmp_dev, *S_last_level_dev;
				cudaMalloc((void**)&S_tmp_dev, last_size * sizeof(unsigned));
				//std::cout << "last_size:" << last_size * sizeof(unsigned) << std::endl;
				//unsigned* dev_a, * dev_b, * dev_c;
				//cudaMalloc((void**)&dev_a, A.size() * sizeof(unsigned));
				
				//checkCudaErrors(cudaGetLastError());

				cudaMalloc((void**)&S_last_level_dev, last_size * sizeof(unsigned));
				//checkCudaErrors(cudaGetLastError());
				cudaMemcpy(S_last_level_dev, &S[level - 1][0], last_size * sizeof(unsigned), cudaMemcpyHostToDevice);
				//checkCudaErrors(cudaGetLastError());
				auto start1 = std::chrono::high_resolution_clock::now();
				Intersect1 << <64, 128 >> > (row_dev, col_dev, vertex - 1, S_last_level_dev, S_tmp_dev, L.vertices[vertex - 1].neighbor.size(), S[level - 1].size());
				auto end1 = std::chrono::high_resolution_clock::now();
				//InterSection(L.vertices[vertex - 1].neighbor, S[level - 1], S[level]);
				//checkCudaErrors(cudaGetLastError());
				//cudaDeviceSynchronize();
				//cudaDeviceReset();
				//checkCudaErrors(cudaGetLastError());
				cudaMemcpy(S_tmp, S_tmp_dev, last_size * sizeof(unsigned), cudaMemcpyDeviceToHost);
				//checkCudaErrors(cudaGetLastError());
				S[level].clear();
				for (int l = 0; l < last_size; l++) {
					if (S_tmp[l] != 0) {
						S[level].push_back(S_tmp[l]);
					}
				}
				delete[] S_tmp;
				cudaFree(S_tmp_dev);
				cudaFree(S_last_level_dev);
				//origin
				std::chrono::duration<float> duration1 = end1 - start1;
				STime += duration1.count();
			}
			else {

				//res_list.push_back(std::pair<unsigned, unsigned>(L.vertices[vertex - 1].neighbor.size(), S[level - 1].size()));
				//std::cout << "first size: " << L.vertices[vertex - 1].neighbor.size() << ";second size: " << S[level - 1].size() << std::endl;
				sort(S[level - 1].begin(), S[level - 1].end());
				int last_size = L.vertices[vertex - 1].neighbor.size();
				//std::cout << vertex - 1 << std::endl;
				unsigned* S_tmp_dev, *S_last_level_dev;
				cudaMalloc((void**)&S_tmp_dev, last_size * sizeof(unsigned));
				//checkCudaErrors(cudaGetLastError());
				cudaMalloc((void**)&S_last_level_dev, S[level-1].size() * sizeof(unsigned));
				//checkCudaErrors(cudaGetLastError());
				cudaMemcpy(S_last_level_dev, &S[level - 1][0], S[level - 1].size() * sizeof(unsigned), cudaMemcpyHostToDevice);
				//checkCudaErrors(cudaGetLastError());
				auto start1 = std::chrono::high_resolution_clock::now();
				Intersect2 << <64, 128 >> > (row_dev, col_dev, vertex - 1, S_last_level_dev, S_tmp_dev, L.vertices[vertex - 1].neighbor.size(), S[level - 1].size());
				auto end1 = std::chrono::high_resolution_clock::now();

				//checkCudaErrors(cudaGetLastError());
				//cudaDeviceReset();
				//checkCudaErrors(cudaGetLastError());
				//InterSection(L.vertices[vertex - 1].neighbor, S[level - 1], S[level]);
				unsigned* S_tmp = new unsigned[last_size]();
				cudaMemcpy(S_tmp, S_tmp_dev, last_size * sizeof(unsigned), cudaMemcpyDeviceToHost);
				//checkCudaErrors(cudaGetLastError());
				S[level].clear();
				for (int l = 0; l < last_size; l++) {
					if (S_tmp[l] != 0) {
						//std::cout << "haha";
						S[level].push_back(S_tmp[l]);
					}
				}
				delete[] S_tmp;
				cudaFree(S_tmp_dev);
				//origin
				std::chrono::duration<float> duration1 = end1 - start1;
				STime += duration1.count();
			}
		}
		if (S[level].size() < q || H.vertices[vertex - 1].neighbor.size() < p - level - 1) {
			continue;
		}

		std::vector<unsigned>subH_next;
		if (subH.size() > H.vertices[vertex - 1].neighbor.size()) {
			auto start2 = std::chrono::high_resolution_clock::now();
			InterSection(subH, H.vertices[vertex - 1].neighbor, subH_next);
			auto end2 = std::chrono::high_resolution_clock::now();
			std::chrono::duration<float> duration2 = end2 - start2;
			HTime += duration2.count();
		}
		else {
			auto start2 = std::chrono::high_resolution_clock::now();
			InterSection(H.vertices[vertex - 1].neighbor, subH, subH_next);
			auto end2 = std::chrono::high_resolution_clock::now();
			std::chrono::duration<float> duration2 = end2 - start2;
			HTime += duration2.count();
		}
		//ans.push_back(vertex);
		//std::cout << S[level].size() << ";" << ans.size() << std::endl;
		/*unsigned t[8] = { 30,26,192,37,38,44,42,45 };
		int flag = 0;
		for (auto x : t) {
			auto iter = find(S[level].begin(), S[level].end(), x);
			if (iter == S[level].end()) {
				flag = 1;
			}
		}
		if (flag == 0) {
			for (auto x : ans) {
				std::cout << x << ",";
			}
			std::cout << std::endl;
		}*/
		findCliqueFirstGPU(H, L, p, q, level + 1, S, count, subH_next, STime, HTime, MTime, row_dev, col_dev);
		//ans.pop_back();
	}
}

//CPU baseline
void findClique(Graph& H, Graph& L, int p, int q, int level, std::vector<std::vector<unsigned>>& S, int& count, std::vector<unsigned> subH, float& STime, float& HTime, float& MTime) {

	if (level == p) {
		if (S[level - 1].size() >= q) {
			auto start0 = std::chrono::high_resolution_clock::now();
			count += OrderMul(S[level - 1].size(), q);
			auto end0 = std::chrono::high_resolution_clock::now();
			std::chrono::duration<float> duration0 = end0 - start0;
			MTime += duration0.count();
		}
		return;
	}
	for (auto vertex : subH) {
		if (level == 0) {
			//ans.clear();
			S.clear();

			S.push_back(L.vertices[vertex - 1].neighbor);
		}
		else {

			if (S.size() <= level) {
				S.push_back(std::vector<unsigned>());
			}
			if (L.vertices[vertex - 1].neighbor.size() > S[level - 1].size()) {

				auto start1 = std::chrono::high_resolution_clock::now();
				InterSection(L.vertices[vertex - 1].neighbor, S[level - 1], S[level]);
				auto end1 = std::chrono::high_resolution_clock::now();
				std::chrono::duration<float> duration1 = end1 - start1;
				STime += duration1.count();
			}
			else {

				auto start1 = std::chrono::high_resolution_clock::now();
				InterSection(S[level - 1], L.vertices[vertex - 1].neighbor, S[level]);
				auto end1 = std::chrono::high_resolution_clock::now();
				std::chrono::duration<float> duration1 = end1 - start1;
				STime += duration1.count();
			}
		}
		if (S[level].size() < q || H.vertices[vertex - 1].neighbor.size() < p - level - 1) {
			continue;
		}

		std::vector<unsigned>subH_next;
		if (subH.size() > H.vertices[vertex - 1].neighbor.size()) {
			auto start2 = std::chrono::high_resolution_clock::now();
			InterSection(subH, H.vertices[vertex - 1].neighbor, subH_next);
			auto end2 = std::chrono::high_resolution_clock::now();
			std::chrono::duration<float> duration2 = end2 - start2;
			HTime += duration2.count();
		}
		else {
			auto start2 = std::chrono::high_resolution_clock::now();
			InterSection(H.vertices[vertex - 1].neighbor, subH, subH_next);
			auto end2 = std::chrono::high_resolution_clock::now();
			std::chrono::duration<float> duration2 = end2 - start2;
			HTime += duration2.count();
		}

		findClique(H, L, p, q, level + 1, S, count, subH_next, STime, HTime, MTime);
	}
}

void printGraph(Graph& H) {
	for (int i = 0; i < H.vertex_num; i++) {
		std::cout << i + 1 << ":";
		for (auto v : H.vertices[i].neighbor) {
			std::cout << v << ",";
		}
		std::cout << std::endl;
	}
}


/*void test() {
	/*int a[N], b[N], c[N];
	int* dev_a, * dev_b, * dev_c;

	cudaMalloc((void**)&dev_a, N * sizeof(int));
	cudaMalloc((void**)&dev_b, N * sizeof(int));
	cudaMalloc((void**)&dev_c, N * sizeof(int));

	for (int i = 0; i < N; i++) {
		a[i] = -i;
		b[i] = i * i;
	}

	cudaMemcpy(dev_a, a, N * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, b, N * sizeof(int), cudaMemcpyHostToDevice);

	add << <N, 1 >> > (dev_a, dev_b, dev_c);

	cudaMemcpy(c, dev_c, N * sizeof(int), cudaMemcpyDeviceToHost);

	for (int i = 0; i < N; i++) {
		printf("%d + %d = %d\n", a[i], b[i], c[i]);
	}

	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);

	return 0;*/

	/*std::vector<unsigned> A { 1,4,11,12,20,21,22,75,98,101 };
	std::vector<unsigned> B{ 11,21,102 };
	
	unsigned* dev_a, * dev_b, * dev_c;

	cudaMalloc((void**)&dev_a, A.size() * sizeof(unsigned));
	cudaMalloc((void**)&dev_b, B.size() * sizeof(unsigned));
	cudaMalloc((void**)&dev_c, B.size() * sizeof(unsigned));


	cudaMemcpy(dev_a, &A[0], 10 * sizeof(unsigned), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, &B[0], 3 * sizeof(unsigned), cudaMemcpyHostToDevice);

	Intersect << <1, 3 >> > (dev_a, dev_b, dev_c, 10, 3);

	unsigned* C = new unsigned[B.size()]();
	cudaMemcpy(C, dev_c, B.size() * sizeof(unsigned), cudaMemcpyDeviceToHost);

	for (int i = 0; i < 3; i++) {
		printf("%d\n", C[i]);
	}

	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);
	delete[]C;*/

void test(int p, int q) {
	//int p = 12, q = 8;
	CSR csr;
	Graph graphL;
	readFile(graphL, false);
	std::cout << "vertexNum:" << graphL.vertex_num << "; edgeNum:" << graphL.edge_num << std::endl;
	Graph graphR;
	readFile(graphR, true);
	std::cout << "vertexNum:" << graphR.vertex_num << "; edgeNum:" << graphR.edge_num << std::endl;
	Graph graphH;
	Collect2Hop(graphL, graphR, graphH, p);

	std::cout << "vertexNum:" << graphH.vertex_num << "; edgeNum:" << graphH.edge_num << std::endl;
	edgeDirectingByDegree(graphH);
	std::cout << "vertexNum:" << graphH.vertex_num << "; edgeNum:" << graphH.edge_num << std::endl;

	int count = 0;
	std::vector<std::vector<unsigned>>S;
	std::vector<unsigned>subH;
	for (int i = 0; i < graphH.vertex_num; i++) {
		subH.push_back(i + 1);
	}

	float interSTime = 0, interHTime = 0, MTime = 0;
	auto start = std::chrono::high_resolution_clock::now();

	findClique(graphH, graphL, q, p, 0, S, count, subH, interSTime, interHTime, MTime);

	auto end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<float> duration = end - start;
	std::cout << "All time: " << duration.count() << "s" << std::endl;
	std::cout << "S Time: " << interSTime << "s; H Time: " << interHTime << "s; M Time: " << MTime << "s" << std::endl;
	
	std::cout << count << std::endl;
}

unsigned long long calBits(int num) {
	unsigned long long count = 0;
	while (num != 0) {
		num >>= 1;
		count++;
	}
	return count;
}

bool cmpWorkload(std::pair<unsigned long long, unsigned> a, std::pair<unsigned long long, unsigned> b) {
	return a.first > b.first;
}

void calWork(Graph& direct, Graph& L, std::vector<unsigned long long>& workload) {
	std::vector<int> non_zero_vertex;
	for (int i = 0; i < direct.vertex_num; i++) {
		if (direct.vertices[i].neighbor.size() == 0) {
			workload[i] = calBits(L.vertices[i].neighbor.size()) + calBits(direct.vertices[i].neighbor.size());
		}
		else {
			non_zero_vertex.push_back(i);
		}
	}
	int flag1 = 1;
	int count = 0;
	do {
		flag1 = 1;
		for (int j = 0; j < non_zero_vertex.size(); j++) {
			//std::cout << j << std::endl;
			unsigned long long workload_tmp = 0;
			int flag = 0;
			if (non_zero_vertex[j] != -1) {
				flag1 = 0;
				for (auto nei : direct.vertices[non_zero_vertex[j]].neighbor) {
					if (workload[nei-1] == 0) {
						flag = 1;
						break;
					}
					workload_tmp += workload[nei - 1];
				}
				//workload_tmp += L.vertices[non_zero_vertex[j]].neighbor.size() / 10;
				if (flag == 0) {
					workload[non_zero_vertex[j]] = workload_tmp + calBits(L.vertices[non_zero_vertex[j]].neighbor.size()) + calBits(direct.vertices[non_zero_vertex[j]].neighbor.size());
					count++;
					//std::cout << workload_tmp << "; now: "<< count <<std::endl;
					non_zero_vertex[j] = -1;
				}
			}
			/*if (count == 177 && j == 300) {
				printf("H\n");
			}*/
		}
		/*if (count == 177) {
			for (int i = 0; i < non_zero_vertex.size(); i++) {
				if (non_zero_vertex[i] != -1) {
					std::cout << non_zero_vertex[i] << ":[";
					for (auto v : direct.vertices[non_zero_vertex[i]].neighbor) {
						if (workload[v] == -1) {
							std::cout << v << ",";
						}
					 }
					std::cout << "]," << std::endl;
				}
			}
			break;
		}*/
	} while (flag1 == 0);
}

void calWorkDFS(Graph& H, Graph& L, std::vector<unsigned long long>& workload, int level, int p, int q, std::vector<unsigned> cand_vertex, unsigned parent, int& count) {
	if (level == p) {
		return;
	}
	for (auto vertex : cand_vertex) {
		//if (level == p) return;
		if (H.vertices[vertex].neighbor.size() < p - level - 1 || L.vertices[vertex].neighbor.size() < q) continue;
		if (level > 0) {
			workload[parent] += (H.vertices[vertex].neighbor.size() + L.vertices[vertex].neighbor.size());
		}
		else {
			parent = vertex;
			if(count>0) std::cout << cand_vertex[count-1] <<": "<<workload[cand_vertex[count - 1]] << std::endl;
			count++;
		}
		std::vector<unsigned> cand_new;
		for (auto v : H.vertices[vertex].neighbor) {
			auto it = find(cand_vertex.begin(), cand_vertex.end(), v - 1);
			if(it != cand_vertex.end()) cand_new.push_back(v - 1);
		}
		if (cand_new.size() < p - level - 1) continue;
		calWorkDFS(H, L, workload, level + 1, p, q, cand_new, parent,count);
	}
}

int main() {
	//test(4, 4);
	int p_list[] = { 22 };
	int q_list[] = { 8 };
	int p, q;

	for (int ii = 0; ii < 1; ii++) {
		for (int jj = 0; jj < 1; jj++) {
			p = p_list[ii], q = q_list[jj];
			CSR csrL;
			CSR csrH;
			Graph graphL;
			readFile(graphL, false);
			/*for (auto node_l : graphL.vertices) {
				sort(node_l.neighbor.begin(), node_l.neighbor.end());
				if (node_l.label == 1) {
					for (auto x : node_l.neighbor) {
						std::cout << x << ",";
					}
				}
			}*/
			for (int i = 0; i < graphL.vertex_num; i++) {
				sort(graphL.vertices[i].neighbor.begin(), graphL.vertices[i].neighbor.end());
			}
			std::cout << "vertexNum:" << graphL.vertex_num << "; edgeNum:" << graphL.edge_num << std::endl;
			
			Graph graphR;
			readFile(graphR, true);
			for (int i = 0; i < graphR.vertex_num; i++) {
				sort(graphR.vertices[i].neighbor.begin(), graphR.vertices[i].neighbor.end());
			}
			std::cout << "vertexNum:" << graphR.vertex_num << "; edgeNum:" << graphR.edge_num << std::endl;
			Graph graphH;
			Collect2Hop(graphL, graphR, graphH, p);
			
			/*int count = 0;
			for (int i = 0; i < graphH.vertex_num; i++) {
				for (auto v : graphH.vertices[i].neighbor) {
					if (v==1) {
						count++;
					}
				}
			}
			for (auto x : graphH.vertices[0].neighbor) {
				std::cout << x << ",";
			}
			std::cout << std::endl;
			std::cout << count << ";" << graphH.vertices[0].neighbor.size() << std::endl;*/
			/*for (auto v : graphH.vertices[6947].neighbor) {
				std::cout << v << ",";
			}
			std::cout << std::endl;*/

			std::cout << "vertexNum:" << graphH.vertex_num << "; edgeNum:" << graphH.edge_num << std::endl;
			edgeDirectingByDegree(graphH);
			std::cout << "vertexNum:" << graphH.vertex_num << "; edgeNum:" << graphH.edge_num << std::endl;

			int zero_count = 0;
			std::vector<unsigned> nonzerovertex;
			for (int i = 0; i < graphH.vertex_num; i++) {
				if (graphH.vertices[i].neighbor.size() < q || graphL.vertices[i].neighbor.size() < p) {
					zero_count++;
				}
				else {
					nonzerovertex.push_back(i);
				}
			}

			/*for (auto x : graphH.vertices[nonzerovertex[0]].neighbor) {
				std::cout << x << ",";
			}*/
			/*for (auto v : graphH.vertices[6947].neighbor) {
				std::cout << v << ",";
			}
			std::cout << std::endl;*/

			//std::cout << "VertexNum:" << zero_count << std::endl;
			//std::vector<unsigned long long>workload_tmp;
			//for (int i = 0; i < graphH.vertex_num; i++) {
			//	workload_tmp.push_back(0);
			//}
			//int countc = 0;
			//calWorkDFS(graphH, graphL, workload_tmp, 0, q, p, nonzerovertex, 0,countc);
			///*for (int i = 0; i < nonzerovertex.size(); i++) {
			//	workload_tmp[nonzerovertex[i]] -= (calBits(graphL.vertices[i].neighbor.size()) + calBits(graphH.vertices[i].neighbor.size()));
			//}*/
			//std::vector<std::pair<unsigned long long, unsigned>> workload;
			//for (int i = 0; i < nonzerovertex.size(); i++) {
			//	workload.push_back(std::pair<unsigned long long, unsigned>(workload_tmp[nonzerovertex[i]], nonzerovertex[i]));
			//}
			//sort(workload.begin(), workload.end(), cmpWorkload);

			//std::vector<unsigned>vertex_sorted;
			//for (auto xxx : workload) {
			//	vertex_sorted.push_back(xxx.second);
			//	std::cout << xxx.first << ":" << xxx.second << std::endl;
			//}

			/*for (int i = 0; i < 3; i++) {
				std::cout << vertex_sorted[i] << "neighbor:";
				for (auto x : graphH.vertices[vertex_sorted[i]].neighbor) {
					std::cout << x << ":" << workload_tmp[x - 1] << ",";
				}
				std::cout << std::endl;
			}*/
			//std::cout << workload[0].first << ":" << workload[0].second << std::endl;
			std::cout << "Number of degree-zero vertex: " << zero_count << "; Others: " << nonzerovertex.size() <<std::endl;

			//nonzerovertex.clear();
			//nonzerovertex.push_back(2108);
			
			//printGraph(graphH);
			graphL.transformToCSR(csrL);
			graphH.transformToCSR(csrH);
			/*for (int j = csrH.row_offset[nonzerovertex[0]]; j < csrH.row_offset[nonzerovertex[0] + 1]; j++) {
				std::cout << csrH.column_index[j] << " ";
			}*/
			/*for (int i = 0; i < graphH.vertex_num; i++) {
				std::cout << i + 1 << ":";
				for (int j = csrH.row_offset[i]; j < csrH.row_offset[i + 1]; j++) {
					std::cout << csrH.column_index[j] << " ";
				}
				std::cout << std::endl;
			}*/
			//printGraph(graphH);

			/*unsigned t[22] = { 116, 770, 2323, 3149, 3407, 3711, 4576, 7841, 9322, 12882, 13760, 19325, 20242, 22435, 31678, 39018, 44685, 46477, 64321, 64577, 93788, 93951 };
			for (auto x : t) {
				std::cout << x << std::endl;
				for (auto y : t) {
					auto it = find(graphH.vertices[x - 1].neighbor.begin(), graphH.vertices[x - 1].neighbor.end(), y);
					if (it != graphH.vertices[x - 1].neighbor.end()) {
						std::cout << y << ",";
					}
				}
				std::cout << std::endl << std::endl;
			}*/

			/*std::vector<std::vector<unsigned>>S;
			std::vector<unsigned>subH;
			for (int i = 0; i < graphH.vertex_num; i++) {
				subH.push_back(i + 1);
			}*/

			//std::vector<std::pair<unsigned, unsigned>> res_list;

			//warm up GPU
			int* warmup = NULL;
			cudaMalloc(&warmup, sizeof(int));
			cudaFree(warmup);
			std::cout << "GPU warmup finished" << std::endl;

			int count = 0;
			int H_size = nonzerovertex.size();
			unsigned* row_offset_dev_L, * column_index_dev_L, * row_offset_dev_H, * column_index_dev_H, * non_vertex_dev;

			cudaMalloc((void**)&row_offset_dev_L, (graphL.vertex_num + 1) * sizeof(unsigned));
			cudaMalloc((void**)&column_index_dev_L, graphL.edge_num * sizeof(unsigned));
			cudaMalloc((void**)&row_offset_dev_H, (graphH.vertex_num + 1) * sizeof(unsigned));
			cudaMalloc((void**)&column_index_dev_H, graphH.edge_num * sizeof(unsigned));
			checkCudaErrors(cudaGetLastError());

			cudaMalloc((void**)&non_vertex_dev, nonzerovertex.size() * sizeof(unsigned));
			cudaMemcpy(non_vertex_dev, &nonzerovertex[0], nonzerovertex.size() * sizeof(unsigned), cudaMemcpyHostToDevice);

			cudaMemcpy(row_offset_dev_L, csrL.row_offset, (graphL.vertex_num + 1) * sizeof(unsigned), cudaMemcpyHostToDevice);
			checkCudaErrors(cudaGetLastError());
			cudaMemcpy(column_index_dev_L, csrL.column_index, graphL.edge_num * sizeof(unsigned), cudaMemcpyHostToDevice);
			checkCudaErrors(cudaGetLastError());

			cudaMemcpy(row_offset_dev_H, csrH.row_offset, (graphH.vertex_num + 1) * sizeof(unsigned), cudaMemcpyHostToDevice);
			checkCudaErrors(cudaGetLastError());
			cudaMemcpy(column_index_dev_H, csrH.column_index, graphH.edge_num * sizeof(unsigned), cudaMemcpyHostToDevice);
			checkCudaErrors(cudaGetLastError());

			int* count_dev, * p_dev, * q_dev, * H_size_dev;
			cudaMalloc((void**)&count_dev, sizeof(int));
			cudaMalloc((void**)&q_dev, sizeof(int));
			cudaMalloc((void**)&H_size_dev, sizeof(int));
			cudaMalloc((void**)&p_dev, sizeof(int));

			cudaMemcpy(count_dev, &count, sizeof(int), cudaMemcpyHostToDevice);
			cudaMemcpy(q_dev, &q, sizeof(int), cudaMemcpyHostToDevice);
			cudaMemcpy(p_dev, &p, sizeof(int), cudaMemcpyHostToDevice);
			cudaMemcpy(H_size_dev, &H_size, sizeof(int), cudaMemcpyHostToDevice);
			checkCudaErrors(cudaGetLastError());

			//clock_t startc, finish;
			//double durationc;
			auto start = std::chrono::high_resolution_clock::now();
			//startc = clock();
			//std::vector<unsigned> ans;
			//int maxsize=0;
			//int max_lev = 0;
			//for (auto x : graphL.vertices[21].neighbor) {
			//	std::cout << x << " ";
			//}
			//std::cout << std::endl;
			////for(int j = 0;j<)
			//for (int i = csr.row_offset[21]; i < csr.row_offset[22]; i++) {
			//	std::cout << csr.column_index[i] << " ";
			//}
			//std::cout << std::endl;
			//findClique(graphH, graphL, q, p, 0, S, count, subH, interSTime, interHTime, MTime, row_offset_dev, column_index_dev);
			findCliqueGPUStre2 << <BLOCKNUM, THREADNUM >> > (row_offset_dev_L, column_index_dev_L, row_offset_dev_H, column_index_dev_H, count_dev, q_dev, p_dev, H_size_dev, non_vertex_dev);
			checkCudaErrors(cudaGetLastError());
			cudaMemcpy(&count, count_dev, sizeof(int), cudaMemcpyDeviceToHost);
			checkCudaErrors(cudaGetLastError());
			//std::cout << max_lev << std::endl;

			auto end = std::chrono::high_resolution_clock::now();
			//finish = clock();
			//durationc = ((double)(finish - startc) / CLOCKS_PER_SEC);
			std::chrono::duration<float> duration = end - start;
			std::cout << "All time: " << duration.count() << "s" << std::endl;
			//std::cout << durationc << std::endl;
			cudaFree(non_vertex_dev);

			cudaFree(row_offset_dev_L);
			cudaFree(column_index_dev_L);
			cudaFree(row_offset_dev_H);
			cudaFree(column_index_dev_H);
			cudaFree(count_dev);
			cudaFree(p_dev);
			cudaFree(q_dev);
			cudaFree(H_size_dev);
			//std::cout << maxsize << std::endl;
			//std::cout << res_list.size() << std::endl;
			/*FILE* fp = NULL;
			fopen_s(&fp, "res.txt", "w");
			for (auto x : res_list) {
				fprintf(fp,"%u %u\n", x.first,x.second);
			}
			fclose(fp);*/
			/*graphH.transformToCSR(csr);
			for (int i = 0; i < 10; i++) {
				std::cout << graphH.csr->row_offset[i] << std::endl;
			}*/


			std::cout << "The number of ("<< p << "," << q <<")-biclique is " << count << std::endl;
		}
	}
	//test();
	return 0;
}

//int main() {
//	test(12, 8);
//	return 0;
//}