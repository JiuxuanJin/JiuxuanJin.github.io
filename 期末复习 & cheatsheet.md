<h1 align='center'> Huffman 编码</h1>

每次取权值最小的两棵子树，合并它们，权值相加放回队列。

```python
import heapq
from typing import Dict, Optional

class HuffmanNode:
    """Huffman 树节点"""
    def __init__(self, char: Optional[str], weight: int):
        self.char = char          # 字符（叶子节点有值，内部节点为 None）
        self.weight = weight      # 权值（频率）
        self.left = None          # 左子树
        self.right = None         # 右子树

    # 定义小于运算符，用于最小堆比较（heapq 根据此比较节点）
    def __lt__(self, other):
        return self.weight < other.weight

def build_huffman_tree(freq_map: Dict[str, int]) -> HuffmanNode:

    # 1. 为每个字符创建叶子节点，并全部放入最小堆
    min_heap = []
    for char, weight in freq_map.items():
        node = HuffmanNode(char, weight)
        heapq.heappush(min_heap, node)

    # 2. 当堆中多于1个节点时，反复合并最小的两个
    while len(min_heap) > 1:
        # 取出权值最小的两个节点（子树）
        left = heapq.heappop(min_heap)   # 最小的作为左子树
        right = heapq.heappop(min_heap)  # 第二小的作为右子树

        # 合并：创建新内部节点，权值为两子树权值之和，字符为 None
        merged = HuffmanNode(None, left.weight + right.weight)
        merged.left = left
        merged.right = right

        # 将合并后的新节点放回堆中
        heapq.heappush(min_heap, merged)
        # 此时队列中节点数减少了一个，继续循环

    # 3. 堆中最后剩下的节点即为完整的 Huffman 树的根
    return min_heap[0]


def generate_codes(node: HuffmanNode, current_code: str, code_table: Dict[str, str]):
    """
    递归遍历 Huffman 树，生成每个字符的编码（左0右1）
    """
    if node is None:
        return

    # 叶子节点：记录该字符对应的编码
    if node.char is not None:
        code_table[node.char] = current_code
        return

    # 非叶子节点：向左走加 '0'，向右走加 '1'
    generate_codes(node.left, current_code + "0", code_table)
    generate_codes(node.right, current_code + "1", code_table)


def huffman_encoding(text: str) -> Dict[str, str]:
    """对输入文本进行 Huffman 编码，返回字符-编码映射表"""
    if not text:
        return {}

    # 1. 统计每个字符出现的频率
    freq_map = {}
    for ch in text:
        freq_map[ch] = freq_map.get(ch, 0) + 1

    # 2. 构建 Huffman 树（每次取权值最小的两棵子树合并）
    root = build_huffman_tree(freq_map)

    # 3. 从根开始生成编码表
    code_table = {}
    generate_codes(root, "", code_table)

    return code_table
```

<h1 align='center'> 二叉堆</h1>

主要性质：
1. **结构性质**：是一棵完全二叉树，除最后一层外所有层都被填满，最后一层节点从左到右填入。
2. **堆序性质**：满足父节点与子节点的大小关系（最小或最大）。
由于完全二叉树适合数组存储，对于索引 `i` 的节点（根节点为 `0`）：
- 父节点索引：`(i - 1) // 2`
- 左子节点索引：`2 * i + 1`
- 右子节点索引：`2 * i + 2`
**核心操作及复杂度**：
- `peek`：获取极值 O(1)
- `push`：插入元素，末尾添加后**上浮** O(log n)
- `pop`：删除极值，用末尾元素替换根后**下沉** O(log n)
- `heapify`：从无序数组构建堆，自底向上下沉 O(n)

```python
from typing import List, Optional

class MinHeap:
    def __init__(self, data: Optional[List[int]] = None):
        """
        初始化最小堆
        :param data: 可选，初始数据列表，会调用 heapify 构建堆
        """
        if data is None:
            self.heap = []
        else:
            # 复制一份数据，避免外部修改影响内部数组
            self.heap = data[:]
            # 自底向上进行堆化，时间复杂度 O(n)
            self._build_heap()

    def _parent(self, idx: int) -> int:
        """返回索引 idx 的父节点索引"""
        return (idx - 1) // 2

    def _left_child(self, idx: int) -> int:
        """返回索引 idx 的左子节点索引"""
        return 2 * idx + 1

    def _right_child(self, idx: int) -> int:
        """返回索引 idx 的右子节点索引"""
        return 2 * idx + 2

    def _swap(self, i: int, j: int) -> None:
        """交换堆中索引 i 和 j 的元素"""
        self.heap[i], self.heap[j] = self.heap[j], self.heap[i]

    def _sift_up(self, idx: int) -> None:
        """
        上浮操作：将索引 idx 的元素沿着树向上移动，直到满足堆序
        用于插入新元素后恢复堆的性质
        """
        # 当 idx 不是根节点（索引 > 0）且当前节点小于父节点时，交换并继续上浮
        while idx > 0:
            parent = self._parent(idx)
            if self.heap[idx] < self.heap[parent]:
                self._swap(idx, parent)
                idx = parent  # 更新索引，继续向上比较
            else:
                # 已满足堆序，停止上浮
                break

    def _sift_down(self, idx: int) -> None:
        """
        下沉操作：将索引 idx 的元素沿着树向下移动，直到满足堆序
        用于删除堆顶后或构建堆时恢复性质
        """
        size = len(self.heap)
        while True:
            left = self._left_child(idx)
            right = self._right_child(idx)
            smallest = idx  # 假设当前节点是最小的

            # 如果左子节点存在且比当前最小值还小，则更新最小值索引
            if left < size and self.heap[left] < self.heap[smallest]:
                smallest = left

            # 如果右子节点存在且比当前最小值还小，则更新最小值索引
            if right < size and self.heap[right] < self.heap[smallest]:
                smallest = right

            # 若最小值索引不是当前节点，说明需要交换，并继续下沉
            if smallest != idx:
                self._swap(idx, smallest)
                idx = smallest  # 更新索引，继续向下比较
            else:
                # 已到达正确位置，停止下沉
                break

    def _build_heap(self) -> None:
        """
        从无序数组构建最小堆 (heapify)
        从最后一个非叶子节点开始，依次向前执行下沉操作
        时间复杂度 O(n)
        """
        if not self.heap:
            return
        # 最后一个非叶子节点的索引为 len(heap)//2 - 1
        start = len(self.heap) // 2 - 1
        for i in range(start, -1, -1):
            self._sift_down(i)

    def push(self, value: int) -> None:
        """
        插入一个元素到堆中
        步骤：添加到末尾，然后上浮到合适位置
        时间复杂度 O(log n)
        """
        self.heap.append(value)      # 将新元素放在末尾
        self._sift_up(len(self.heap) - 1)  # 从末尾开始上浮

    def pop(self) -> Optional[int]:
        """
        删除并返回堆顶（最小值）
        步骤：取根元素，将最后一个元素移到根，然后下沉
        时间复杂度 O(log n)
        """
        if not self.heap:
            return None  # 堆为空时返回 None
        if len(self.heap) == 1:
            return self.heap.pop()  # 只有一个元素时直接弹出

        root = self.heap[0]                 # 保存堆顶（最小值）
        # 将最后一个元素移到堆顶，并删除最后一个元素
        self.heap[0] = self.heap.pop()
        self._sift_down(0)                  # 从根开始下沉，恢复堆序
        return root

    def peek(self) -> Optional[int]:
        """返回堆顶（最小值）但不删除，O(1)"""
        return self.heap[0] if self.heap else None

    def size(self) -> int:
        """返回堆中元素个数"""
        return len(self.heap)

    def is_empty(self) -> bool:
        """判断堆是否为空"""
        return len(self.heap) == 0

    def __str__(self) -> str:
        """打印堆的列表表示（层序遍历顺序）"""
        return str(self.heap)
```

<h1 align='center'> 二叉搜索树</h1>

二叉搜索树（Binary Search Tree，BST），它是映射的另一种实现。我们感兴趣的不是元素在树中的确切位置，而是如何利用二叉树结构提供高效的搜索。
二叉搜索树依赖于这样一个性质：小于父节点的键都在左子树中，大于父节点的键则都在右子树中。我们称这个性质为二叉搜索性。利用**二叉搜索树（BST）的一个核心性质**：**中序遍历（Inorder）结果是升序排列的。

```python
class TreeNode:
    """二叉搜索树节点"""
    def __init__(self, key):
        self.key = key          # 节点键值
        self.left = None        # 左子节点
        self.right = None       # 右子节点

class BST:
    def __init__(self):
        self.root = None
    # ------------------------- 插入操作 -------------------------
    def insert(self, key):
        """向BST中插入一个新节点（公开接口）"""
        self.root = self._insert_recursive(self.root, key)

    def _insert_recursive(self, node, key):
        """
        参数：
            node: 当前子树的根节点
            key: 要插入的键值
        返回：
            插入后子树的根节点
        """
        # 基本情况：找到空位，创建并返回新节点
        if node is None:
            return TreeNode(key)

        # 递归情况：根据BST性质决定向左还是向右
        if key < node.key:
            # 插入到左子树，并更新左子节点指针
            node.left = self._insert_recursive(node.left, key)
        elif key > node.key:
            # 插入到右子树，并更新右子节点指针
            node.right = self._insert_recursive(node.right, key)
        # 若key相等，通常BST不允许重复值，这里默认忽略（也可按需求处理）
        # 返回当前节点（可能未变，但左右子树可能已更新）
        return node

    # ------------------------- 删除操作 -------------------------
    def delete(self, key):
        """从BST中删除指定键值的节点（公开接口）"""
        self.root = self._delete_recursive(self.root, key)

    def _delete_recursive(self, node, key):
        """
        参数：
            node: 当前子树的根节点
            key: 要删除的键值
        返回：
            删除后子树的根节点（可能被替换）
        """
        # 基本情况：空树或未找到节点，直接返回None
        if node is None:
            return None

        # 在左子树中查找并删除
        if key < node.key:
            node.left = self._delete_recursive(node.left, key)
        # 在右子树中查找并删除
        elif key > node.key:
            node.right = self._delete_recursive(node.right, key)
        else:
            # 找到了要删除的节点 (node)
            
            # 情况1：节点只有一个子节点或没有子节点
            if node.left is None:
                # 只有右子节点（或为None），用右子节点替换当前节点
                return node.right
            elif node.right is None:
                # 只有左子节点，用左子节点替换当前节点
                return node.left

            # 情况2：节点有两个子节点
            # 找到右子树中的最小值节点（即中序后继）
            successor = self._find_min(node.right)
            # 用后继节点的值替换当前节点的值
            node.key = successor.key
            # 递归删除右子树中的后继节点（该后继节点一定没有左子节点）
            node.right = self._delete_recursive(node.right, successor.key)

        # 返回当前节点（可能已修改子树结构）
        return node
```
- **插入**：从根节点出发，遇到空位置就创建新节点；通过递归更新路径上的指针。
- **删除**：
  1. 叶子节点：直接移除（返回 `None`）。
  2. 单子节点：用唯一的子节点替代当前节点。
  3. 双子节点：用**中序后继**（右子树的最小值）覆盖删除节点的值，然后递归删除后继节点。

<h1 align='center'> 并查集</h1>

并查集（Disjoint Set Union，简称 DSU 或 Union-Find）是一种树型数据结构，用于处理一些**不相交集合**的合并及查询问题。它主要支持两种操作：
- **Find**：查找元素属于哪个集合（即找到所在树的根节点）。
- **Union**：将两个元素所在的集合合并为一个集合。
通常引入两种优化：
1. **路径压缩**：在 `find` 过程中，将查找路径上的所有节点直接连接到根节点，使后续查找变快。
2. **按秩合并**：在 `union` 时，总是将秩（大致为树高）较小的树合并到秩较大的树上，避免树过高。

```python
class UnionFind:
    def __init__(self, n: int):
        """
        初始化 n 个互不相交的集合，元素编号为 0 到 n-1。
        每个元素初始时自成一个集合。
        """
        # parent[i] 表示元素 i 的父节点，初始时每个元素的父节点是自己
        self.parent = list(range(n))
        # rank[i] 表示以 i 为根的树的近似高度（秩）
        # 初始时每棵树只有一个节点，秩为 0
        self.rank = [0] * n
        # 集合的数量，初始时有 n 个独立集合
        self.count = n

    def find(self, x: int) -> int:
        """
        查找元素 x 所在集合的根节点（代表元），同时进行路径压缩。
        路径压缩：
            在递归查找的过程中，将路径上的每个节点直接连接到根节点，
            从而极大缩短后续查找的路径长度。
        """
        # 如果 x 的父节点不是自己，说明 x 不是根节点
        if self.parent[x] != x:
            # 递归查找根节点，并将 x 的父节点直接设置为根节点（路径压缩）
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x: int, y: int) -> bool:
        """
        合并元素 x 和元素 y 所在的集合。
        按秩合并：
            比较两棵树的秩，将秩较小的树连接到秩较大的树上。
            如果两棵树秩相同，则任选一棵作为新根，并将其秩加 1。
        如果 x 和 y 原本属于不同集合，则合并并返回 True；
        如果它们已经在同一集合中，则不操作并返回 False。
        """
        # 找到 x 和 y 的根节点
        root_x = self.find(x)
        root_y = self.find(y)

        # 如果根节点相同，说明 x 和 y 已在同一集合中，无需合并
        if root_x == root_y:
            return False

        # 按秩合并：将秩较小的树挂到秩较大的树上
        if self.rank[root_x] < self.rank[root_y]:
            self.parent[root_x] = root_y
        elif self.rank[root_x] > self.rank[root_y]:
            self.parent[root_y] = root_x
        else:
            # 秩相等时，任意挂载，并将新根的秩增加 1
            self.parent[root_y] = root_x
            self.rank[root_x] += 1

        # 合并后集合总数减少 1
        self.count -= 1
        return True

    def connected(self, x: int, y: int) -> bool:
        """
        判断元素 x 和元素 y 是否属于同一个集合。
        """
        return self.find(x) == self.find(y)

    def get_count(self) -> int:
        """
        返回当前并查集中的不相交集合数量。
        """
        return self.count
```

<h1 align='center'> 拓扑排序</h1>

### 1.1 Kahn 算法（BFS）

Kahn算法是基于广度优先搜索（BFS）的一种拓扑排序算法。Kahn算法的基本思想是通过不断地移除图中的入度为0的顶点，并将其添加到拓扑排序的结果中，直到图中所有的顶点都被移除。
1. 统计所有顶点的入度。
2. 将入度为 0 的顶点入队。
3. 弹出队首 u，将其加入结果集，并将其邻居的入度减 1。
4. 若邻居入度变为 0，则入队。
5. 重复步骤3，直到队列为空。
6. **判定**：若结果集顶点数小于原图顶点数，说明图中存在**环**。

```python
from collections import deque, defaultdict

def kahn_topological_sort(num_vertices, edges):
    """
    参数:
        num_vertices: 图中顶点的数量（顶点编号假设为 0 ~ num_vertices-1）
        edges: 有向边的列表，每条边为 (u, v) 表示从 u 指向 v
    返回:
        拓扑排序结果列表（如果存在）
        若图中存在环，返回 None
    """
    # 1. 构建邻接表并统计所有顶点的入度
    adj = defaultdict(list)  # 邻接表，存储每个顶点的所有后继
    indegree = [0] * num_vertices  # 入度数组，初始化全0
    
    for u, v in edges:
        adj[u].append(v)      # 添加有向边 u -> v
        indegree[v] += 1      # v 的入度加1

    # 2. 将所有入度为 0 的顶点加入队列
    queue = deque([i for i in range(num_vertices) if indegree[i] == 0])
    topo_order = []  # 存放最终的拓扑序列

    # 3. 不断弹出队列中的顶点进行处理
    while queue:
        u = queue.popleft()        # 弹出队首顶点
        topo_order.append(u)       # 将其加入结果序列

        # 遍历 u 的所有邻居，将这些邻居的入度减 1
        for v in adj[u]:
            indegree[v] -= 1
            # 4. 若邻居的入度变为 0，则加入队列
            if indegree[v] == 0:
                queue.append(v)

    # 6. 判定：若结果序列中的顶点数小于原图顶点数，说明存在环
    if len(topo_order) < num_vertices:
        return None   # 存在环，无法完成拓扑排序
    else:
        return topo_order
```

### 1.2 DFS 拓扑序列

对图进行深度优先搜索，计算每个顶点的“结束时间”（Finish Time）。将顶点按结束时间**递减顺序**排列，即得到拓扑序列.

```python
def dfs_topological_sort(num_vertices, edges):
    """
    参数:
        num_vertices: 图中顶点的数量
        edges: 有向边的列表，每条边为 (u, v) 表示从 u 指向 v
    
    返回:
        拓扑排序结果列表（如果存在）
        若图中存在环，返回 None
    """
    # 构建邻接表
    adj = defaultdict(list)
    for u, v in edges:
        adj[u].append(v)

    # 记录每个顶点的状态: 0 = 未访问, 1 = 正在访问（在当前递归栈中）, 2 = 已完成
    state = [0] * num_vertices
    finish_order = []  # 按照结束时间从早到晚排列（回溯时添加）
    has_cycle = False  # 是否检测到环

    def dfs(node):
        nonlocal has_cycle
        if has_cycle:    # 如果已经发现环，提前结束
            return
        
        state[node] = 1  # 标记为“正在访问”

        for neighbor in adj[node]:
            if state[neighbor] == 0:
                dfs(neighbor)           # 未访问的邻居，递归深入
            elif state[neighbor] == 1:
                # 发现正在访问的顶点，说明存在后向边，即存在环
                has_cycle = True
                return

        # 所有邻居处理完毕，当前顶点访问完成
        state[node] = 2
        finish_order.append(node)   # 回溯时记录结束时间（实际上就是后序）

    # 对每个未访问的顶点启动 DFS（处理非连通图的情况）
    for v in range(num_vertices):
        if state[v] == 0:
            dfs(v)
            if has_cycle:
                return None   # 检测到环，直接返回

    # 将结束时间递减排列（即逆序 finish_order）得到拓扑序列
    topo_order = finish_order[::-1]
    return topo_order
```

<h1 align='center'> 环检测</h1>
### 2.1 无向图中判断是否有环

#### 并查集（Union-Find）
并查集算法之所以可以用来检测无向图中是否存在环，是因为它能够高效地维护和查询图中节点之间的连通性。在构建图的过程中，如果两个节点已经属于同一个连通分量（即它们的根相同），那么再添加一条连接这两个节点的边就会形成一个环。
- 初始每个点属于不同的集合。
- 每条边连接两个点，如果两个点已经在一个集合中，说明成环。
#### DFS (Parent指针)
这是最常见的方法。
- 使用 DFS（深度优先搜索）遍历图。
- 每次 DFS 时，记录当前节点的“父亲节点”。
- 如果访问到了已经访问过的节点，且不是当前节点的父亲节点，说明存在环。
### 2.2 有向图中判断是否有环
#### Kahn 算法
是实现拓扑排序的一种具体算法，它通过管理节点的入度来完成排序，并且可以同时用于检测图中是否存在环（如果最终排序结果中的节点数量少于图中节点总数，则说明图中存在环）。
#### DFS (三色标记法)
**DFS 拓扑排序与判环原理**，DFS 拓扑排序的核心是给节点标记 **三种状态**（也称“三色标记法”）：
1. **未访问 (Unvisited)**：还没碰到这个点。
2. **访问中 (Visiting)**：正在以这个点为起点进行 DFS，还在它的递归栈里。
3. **已完成 (Visited/Finished)**：这个点及其所有子孙节点都已经探索完毕。
**判定准则：**
- 如果 DFS 遇到了一个“**访问中**”的节点 → **存在环**（Back Edge）。
- 拓扑排序结果：节点进入“**已完成**”状态的顺序的**逆序**。

<h1 align='center'>强连通分量 SCC</h1>
### 1. 定义
在有向图 \( G = (V, E) \) 中，如果存在一条从顶点 \( u \) 到 \( v \) 的路径，同时也存在一条从 \( v \) 到 \( u \) 的路径，则称 \( u \) 和 \( v \) **强连通**。  
**强连通分量** 是极大的强连通顶点子集：子集内任意两点强连通，且加上任何其他顶点都会破坏该性质。
**直观理解**：SCC 是图中能互相到达的“强连通团”，缩成一个点后得到 DAG（有向无环图）。

### 2. 核心概念
- **缩点（Condensation Graph）**：将每个 SCC 缩成一个超级节点，新图一定是 **DAG**。
- **出度/入度**：在 DAG 上，入度为 0 的 SCC 无来自其他分量的边，出度为 0 的 SCC 无指向其他分量的边。
- 
### 3. 常见算法

#### **3.1 Kosaraju 算法**（两次 DFS，简洁直观）
**步骤**：
1. 对原图执行 DFS，记录**完成时间**（后序），将顶点压入栈（完成时间晚的在上）。
2. 构造原图的**反向图**（所有边反向）。
3. 从栈顶依次弹出顶点，在反向图上进行 DFS，每次 DFS 访问到的顶点构成一个 SCC。
**复杂度**：O(V + E)，两次遍历。
**为什么有效**：  
第一次 DFS 的结束顺序保证了在反向图中，从结束时间最晚的顶点出发能完整遍历其所在 SCC，且不会混入其他 SCC。

```python
from collections import defaultdict
from typing import List, Dict, Set

def kosaraju_scc(graph: Dict[int, List[int]]) -> List[List[int]]:
    """
    使用 Kosaraju 算法寻找有向图中的所有强连通分量 (SCC)。
    参数:
        graph: 邻接表形式的有向图，键为节点，值为邻居列表。
               例如 {1: [2], 2: [3], 3: [1]}。
    返回:
        scc_list: 列表的列表，每个内部列表包含一个 SCC 中的节点。
                  返回顺序是缩点后 DAG 的逆拓扑序 (即出度为 0 的 SCC 最先被输出)。
    """
    # 1. 收集所有节点，确保孤立节点也能被处理。
    all_nodes = set(graph.keys())
    for neighbors in graph.values():
        all_nodes.update(neighbors)
    
    # 2. 第一步：正向图的 DFS，记录完成顺序。
    visited = set()
    finish_stack = []  # 完成时间栈：后进先出，最后完成的节点在栈顶。
    
    def dfs_forward(u: int):
        visited.add(u)
        for v in graph.get(u, []):
            if v not in visited:
                dfs_forward(v)
        # 当前节点的所有邻居都探索完毕，记录它（后序）。
        finish_stack.append(u)
    
    # 外层循环保证即使图不连通，所有节点都会被处理。
    for node in all_nodes:
        if node not in visited:
            dfs_forward(node)
    
    # 3. 构建反向图（转置图）。
    reversed_graph = defaultdict(list)
    for u in graph:
        for v in graph[u]:
            reversed_graph[v].append(u)
    # 对于可能没有出边的节点，也要在反向图中占个位置。
    for node in all_nodes:
        if node not in reversed_graph:
            reversed_graph[node] = []
    
    # 4. 第二步：按完成时间从晚到早，在反向图上 DFS 收集 SCC。
    visited.clear()
    scc_list = []
    
    def dfs_collect(u: int, component: List[int]):
        visited.add(u)
        component.append(u)
        for v in reversed_graph[u]:
            if v not in visited:
                dfs_collect(v, component)
    
    # 从栈顶依次弹出节点（即最后完成的节点最先被处理）。
    while finish_stack:
        node = finish_stack.pop()
        if node not in visited:
            component = []
            dfs_collect(node, component)
            scc_list.append(component)
    
    return scc_list
```

#### **3.2 Tarjan 算法**（单次 DFS，更高效）
**核心变量**：
- `dfn[u]`：顶点 u 的 DFS 序号（时间戳）
- `low[u]`：u 及其后代能追溯到的**最小的 dfn**
- 使用栈维护当前 DFS 路径上的顶点。
**判定**：若 `dfn[u] == low[u]`，则 u 是某个 SCC 的根，弹出栈直到 u 构成一个 SCC。

```python
from typing import List, Dict, Set

def tarjan_scc(graph: Dict[int, List[int]]) -> List[List[int]]:
    """
    使用 Tarjan 算法寻找有向图中的所有强连通分量 (SCC)。
    参数:
        graph: 邻接表形式的有向图。
    返回:
        scc_list: 列表的列表，每个内部列表是一个 SCC 的节点集合。
                  返回顺序也是缩点后 DAG 的逆拓扑序。
    """
    # 初始化数据结构
    all_nodes = set(graph.keys())
    for neighbors in graph.values():
        all_nodes.update(neighbors)
    
    dfn = {node: 0 for node in all_nodes}   # 发现时间戳，0 表示未访问
    low = {node: 0 for node in all_nodes}   # 追溯值
    time = 0
    stack = []               # 当前 DFS 路径上待归类的节点
    in_stack = set()         # 快速判断节点是否在栈中
    scc_list = []
    
    def dfs(u: int):
        nonlocal time
        # 1. 分配时间戳，并入栈
        time += 1
        dfn[u] = low[u] = time
        stack.append(u)
        in_stack.add(u)
        
        # 2. 遍历所有邻居
        for v in graph.get(u, []):
            if dfn[v] == 0:  # 树边：v 未被访问过
                dfs(v)
                low[u] = min(low[u], low[v])   # 孩子能到的早期节点，u 也能到
            elif v in in_stack:  # 回边或横叉边指向栈内节点
                # 关键：这里必须用 dfn[v] 而不是 low[v]，
                # 因为 u→v 这条边本身已经消耗了“唯一一条非树边”的配额。
                low[u] = min(low[u], dfn[v])
        
        # 3. 判断自己是否是 SCC 的根
        if low[u] == dfn[u]:
            component = []
            while True:
                w = stack.pop()
                in_stack.remove(w)
                component.append(w)
                if w == u:
                    break
            scc_list.append(component)
    
    # 外层循环保证全图节点都被访问
    for node in all_nodes:
        if dfn[node] == 0:
            dfs(node)
    
    return scc_list
```

### 5. 缩点构造 DAG
获得 SCC 后，可建立缩点图：

```python
scc_id = [-1] * n
for idx, comp in enumerate(sccs):
    for v in comp:
        scc_id[v] = idx

dag = [set() for _ in range(len(sccs))]
for u in range(n):
    for v in graph[u]:
        if scc_id[u] != scc_id[v]:
            dag[scc_id[u]].add(scc_id[v])
```
缩点后的图为 **DAG**，可进行拓扑排序、DP 等操作。

<h1 align='center'>最短路径</h1>

| 算法            | 单源/全源 | 负权边 | 负权环 | 时间复杂度                 | 适用图规模     |
|----------------|-----------|--------|--------|----------------------------|----------------|
| Dijkstra       | 单源      | 不可   | 不可   | \( O((V+E)\log V) \)       | 非负权，稀疏/稠密皆可 |
| Bellman-Ford   | 单源      | 可     | 可检测 | \( O(VE) \)                | 可负权，小/中图  |
| Floyd-Warshall | 全源      | 可     | 可检测 | \( O(V^3) \)               | 稠密图，小图    |

## Dijkstra 算法

1. 初始化距离数组 `dist`：源点 0，其余无穷大。
2. 将所有节点放入最小优先队列（按距离排序）。
3. 当队列非空：
   - 取出距离最小的节点 `u`（此时它的最短距离已确定）。
   - 对 `u` 的每条邻边 `(u, v, w)`：
     若 `dist[u] + w < dist[v]`，更新 `dist[v]` 并在队列中调整 `v` 的优先级。
4. 最终 `dist` 即为源点到各点的最短距离。

```python
import heapq

def dijkstra(graph, start):
    """
    Dijkstra 最短路径算法（要求边权非负）
    参数：
        graph: 邻接表表示的图，格式为 dict{节点: list of (邻居, 权重)}
        start: 源点（起点的 key）
    返回：
        dist: 从 start 到各节点的最短距离字典
        prev: 记录每个节点的前驱节点，用于路径重建
    """
    # 初始化距离为无穷大，源点距离为0
    dist = {node: float('inf') for node in graph}
    dist[start] = 0
    
    # 前驱字典：记录如何到达该节点
    prev = {node: None for node in graph}
    
    # 优先队列，元素为 (距离, 节点)
    pq = [(0, start)]
    
    while pq:
        cur_dist, u = heapq.heappop(pq)  # 取出当前距离最小的节点
        # 如果当前取出的距离大于已知最短距离，说明是过时条目，跳过
        if cur_dist > dist[u]:
            continue
        # 遍历 u 的所有邻居 v，权重为 w
        for v, w in graph[u]:
            new_dist = cur_dist + w
            # 松弛操作：找到更短路径则更新
            if new_dist < dist[v]:
                dist[v] = new_dist
                prev[v] = u
                heapq.heappush(pq, (new_dist, v))
    
    return dist, prev
```

##  Bellman-Ford 算法
**算法步骤：**
1. 初始化 `dist[start] = 0`，其余无穷大。
2. 循环 \( |V|-1 \) 次（因为最短路径最多包含 \( |V|-1 \) 条边）：
   - 对图中的每条边 `(u, v, w)`：
     若 `dist[u] + w < dist[v]`，更新 `dist[v] = dist[u] + w`，并记录前驱。
3. 再遍历一遍所有边，若仍存在 `dist[u] + w < dist[v]`，则报告“存在负权环”。

```python
def bellman_ford(graph, start):
    """
    参数：
        graph: 图结构，由于算法需要遍历所有边，直接传入边列表更高效。
               这里采用 (节点, 邻接表) 的输入方式，内部转为边列表。
               或者直接接受边列表：edges = [(u, v, w), ...] 和节点集合。
        start: 源点
    
    返回：
        dist: 距离字典
        prev: 前驱字典
        has_negative_cycle: 布尔值，True 表示存在负权环
    """
    # 提取所有节点和边
    nodes = set(graph.keys())
    edges = []
    for u in graph:
        for v, w in graph[u]:
            edges.append((u, v, w))
            nodes.add(v)  # 确保所有节点都在集合中
    
    # 初始化距离和前驱
    dist = {node: float('inf') for node in nodes}
    dist[start] = 0
    prev = {node: None for node in nodes}
    
    # 进行 |V| - 1 轮松弛
    n = len(nodes)
    for _ in range(n - 1):
        updated = False  # 优化：若本轮无更新，可提前终止
        for u, v, w in edges:
            # 如果 u 可达且经由 u 能更短
            if dist[u] != float('inf') and dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                prev[v] = u
                updated = True
        if not updated:
            break  # 无更新，已经达到最优
    
    # 检测负权环：再遍历一次所有边
    has_negative_cycle = False
    for u, v, w in edges:
        if dist[u] != float('inf') and dist[u] + w < dist[v]:
            has_negative_cycle = True
            break
    
    return dist, prev, has_negative_cycle
```

## 4. Floyd-Warshall 算法

**核心思想：** 动态规划。定义 `dp[k][i][j]` 为从 `i` 到 `j` 只经过编号不超过 `k` 的中间节点的最短距离。状态转移方程：
$dp[k][i][j] = \min(dp[k-1][i][j],\ dp[k-1][i][k] + dp[k-1][k][j])$
空间优化后可用二维矩阵迭代更新：`dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])`。

**算法步骤：**
1. 初始化 `dist[i][j]`：有边则为权重，自己为 0，否则无穷大。
2. 三层循环：外层 `k` 从 0 到 \( V-1 \)，中间 `i`，内层 `j`。
3. 执行状态转移。
4. 检测负权环：若存在 `dist[i][i] < 0`，则存在负权环。

```python
def floyd_warshall(graph):
    """
    参数：
        graph: 可以传入邻接矩阵（二维列表）或者邻接表。
               这里为了通用，假设输入是一个二维矩阵 dist，大小为 V×V，
               其中 dist[i][j] = 边权，若 i==j 为 0，无边为 float('inf')。
               或者接受节点列表和边列表自行构建。
    返回：
        dist: 全源最短距离矩阵
        next_node: 路径重建矩阵，next_node[i][j] 表示 i 到 j 的下一步应走的节点
        has_negative_cycle: 是否含有负权环
    """
    # 示例中使用邻接表构建矩阵，实际可根据需要调整接口
    nodes = list(graph.keys())
    n = len(nodes)
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    
    # 初始化距离矩阵和路径矩阵
    INF = float('inf')
    dist = [[INF] * n for _ in range(n)]
    next_node = [[None] * n for _ in range(n)]
    
    for i in range(n):
        dist[i][i] = 0  # 自己到自己的距离为 0
    
    # 直连边初始化
    for u in graph:
        for v, w in graph[u]:
            i, j = node_to_idx[u], node_to_idx[v]
            dist[i][j] = w
            next_node[i][j] = j
    
    # 动态规划更新
    for k in range(n):
        for i in range(n):
            for j in range(n):
                # 如果经过 k 可以缩短距离
                if dist[i][k] != INF and dist[k][j] != INF:
                    new_dist = dist[i][k] + dist[k][j]
                    if new_dist < dist[i][j]:
                        dist[i][j] = new_dist
                        next_node[i][j] = next_node[i][k]
    
    # 检测负权环
    has_negative_cycle = any(dist[i][i] < 0 for i in range(n))
    
    return dist, next_node, has_negative_cycle, nodes
```


<h1 align='center'> 最小生成树 MST</h1>

在**带权无向连通图**中，一棵生成树是包含所有顶点的树。**最小生成树**是所有生成树中边权**总和最小**的那一棵。
### 1. Prim算法（加点法）

**思路**：
- 任选起点加入“已选集合”
- 每次从**已选集合**到**未选集合**的所有边中，选权值最小的边，把新顶点加入集合
- 重复直到所有顶点加入

```python
import heapq

def prim(n, edges):
    """
    n: 顶点数
    edges: 邻接表形式 [(v, w), ...] 或 边列表需转换
    返回最小总权重
    """
    graph = [[] for _ in range(n)]
    for u, v, w in edges:
        graph[u].append((v, w))
        graph[v].append((u, w))

    visited = [False] * n
    min_heap = [(0, 0)]  # (权重, 顶点)
    total = 0
    cnt = 0

    while min_heap and cnt < n:
        w, u = heapq.heappop(min_heap)
        if visited[u]:
            continue
        visited[u] = True
        total += w
        cnt += 1
        for v, weight in graph[u]:
            if not visited[v]:
                heapq.heappush(min_heap, (weight, v))

    return total
```

### 2. Kruskal算法（加边法）

**思路**：
- 将所有边按权值从小到大排序
- 依次取边，若边的两个顶点**不在同一个连通分量**（不形成环），则加入生成树（并查集维护）
- 重复直到取了 V-1 条边

```python
class DSU:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n
    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x
    def union(self, x, y):
        rx, ry = self.find(x), self.find(y)
        if rx == ry: return False
        if self.rank[rx] < self.rank[ry]:
            self.parent[rx] = ry
        elif self.rank[rx] > self.rank[ry]:
            self.parent[ry] = rx
        else:
            self.parent[ry] = rx
            self.rank[rx] += 1
        return True

def kruskal(n, edges):
    # edges: [(u, v, w)]
    edges.sort(key=lambda e: e[2])
    dsu = DSU(n)
    total = 0
    cnt = 0
    for u, v, w in edges:
        if dsu.union(u, v):
            total += w
            cnt += 1
            if cnt == n - 1:
                break
    return total if cnt == n - 1 else float('inf')
```

### 3. 两种算法对比

| 项目     | Prim            | Kruskal   |
| ------ | --------------- | --------- |
| 数据结构   | 优先队列（堆）         | 并查集 + 排序  |
| 适用图    | 稠密图（点少边多）       | 稀疏图（点多边少） |
| 是否依赖起点 | 是               | 否         |
| 处理负权边  | 可（但最小生成树通常假设正权） | 可         |
| 过程     | 单点扩张            | 森林合并      |

<h1 align='center'> 关键路径 AOE网</h1>
一般来说，AOE网络用来表示一个工程的进行过程，而工程可以分为若干个子工程（即“活动”），显然 AOE 网不能有环，否则会让优先关系出现逻辑错误。因此可以认为AOE网是有向无环图（DAG）。既然AOE网是基于工程提出的概念，那么一定有其需要解决的问题。AOE网需要着重解决两个问题：a.工程起始到终止至少需要多少时间；b.哪条（些）路径上的活动是影响整个工程进度的关键。AOE网络中的最长路径被称为**关键路径**（强调：**关键路径就是AOE网的最长路径**），而把关键路径上的活动称为**关键活动**，显然关键活动会影响整个工程的进度。
基于 AOE 网络的关键路径算法的基本步骤：
> 1. **拓扑序计算 ve (最早发生时间)**：$ve(v) = \max{ve(u) + weight(u, v)}$。
> 2. **逆拓扑序计算 vl (最晚发生时间)**：$vl(u) = \min{vl(v) - weight(u, v)}$。
> 3. **关键活动判定**：若 ve(u) = =vl(v)−weight(u,v)，该活动在关键路径上。

**1. 构建图模型**
首先，构建一个有向无环图（DAG），其中：
- **节点**代表事件或里程碑。
- **边**代表活动，并且每条边有一个权重，表示完成该活动所需的时间。
**2. 计算最早开始时间 (Earliest Start Time, EST)**
使用拓扑排序遍历图，计算每个节点的最早开始时间（EST）。EST 表示从起点到达该节点的最长路径长度。具体步骤如下：
- 初始化所有节点的 EST 为 0。
- 对于图中的每一个节点 `u`，更新其所有邻接节点 `v` 的 EST 值：如果 `EST[u] + weight(u, v)` 大于 `EST[v]`，则更新 `EST[v] = EST[u] + weight(u, v)`。
**3. 计算最晚开始时间 (Latest Start Time, LST)**
反向遍历拓扑排序后的图，计算每个节点的最晚开始时间（LST）。LST 表示为了不延迟整个项目的完成时间，节点 `u` 必须的最晚开始时间。具体步骤如下：
- 初始化终点的 LST 为其 EST 值。
- 对于图中的每一个节点 `u`，更新其所有前置节点 `v` 的 LST 值：如果 `LST[u] - weight(v, u)` 小于 `LST[v]`，则更新 `LST[v] = LST[u] - weight(v, u)`。
**4. 确定关键路径**
- 关键活动是指那些最早开始时间和最晚开始时间相等的活动。即对于边 `(u, v)`，如果 `EST[u] + weight(u, v) == LST[v]`，则 `(u, v)` 是关键活动。
- 通过检查所有边来确定哪些是关键活动，并根据这些关键活动构建关键路径。

<h1 align='center'>一些需要记住的语法、函数、库</h1>
## 1.heapq

`heapq` 是 Python 内置的**最小堆**实现。它能让你用列表来操作堆，保持最小的元素始终在索引 0 的位置。

| 函数                              | 作用           | 时间复杂度      |
| ------------------------------- | ------------ | ---------- |
| `heapq.heappush(heap, item)`    | 插入元素         | O(log n)   |
| `heapq.heappop(heap)`           | 弹出最小值        | O(log n)   |
| `heapq.heappushpop(heap, item)` | 先推入再弹出（效率更高） | O(log n)   |
| `heapq.heapreplace(heap, item)` | 先弹出再推入       | O(log n)   |
| `heapq.heapify(list)`           | 将列表转为堆结构     | O(n)       |
| `heapq.nlargest(k, iterable)`   | 获取最大的 k 个元素  | O(n log k) |
| `heapq.nsmallest(k, iterable)`  | 获取最小的 k 个元素  | O(n log k) |

## 2.集合
### （1） 创建与基本操作

```python
# 创建
s = {1, 2, 3}              # 字面量
s = set([1, 2, 2, 3])      # 从列表（自动去重）→ {1, 2, 3}
s = set()                  # 空集合（{} 是空字典）
# 添加/删除
s.add(4)                   # 添加元素
s.remove(2)                # 删除（不存在报错）
s.discard(5)               # 删除（不存在不报错）
s.pop()                    # 随机删除并返回一个元素
s.clear()                  # 清空
```
### （2） 集合运算

| 运算      | 方法                          | 运算符      | 说明               | 示例（a={1,2,3}, b={2,3,4}） |
| ------- | --------------------------- | -------- | ---------------- | ------------------------ |
| **并集**  | `a.union(b)`                | `a \| b` | 在 a 或 b 中        | `{1,2,3,4}`              |
| **交集**  | `a.intersection(b)`         | `a & b`  | 同时在 a 和 b 中      | `{2,3}`                  |
| **差集**  | `a.difference(b)`           | `a - b`  | 在 a 但不在 b        | `{1}`                    |
| **对称差** | `a.symmetric_difference(b)` | `a ^ b`  | 在其中一个但不同时在       | `{1,4}`                  |
| **子集**  | `a.issubset(b)`             | `a <= b` | a 的所有元素都在 b      | `False`                  |
| **真子集** | -                           | `a < b`  | a 是 b 的子集且 a ≠ b | -                        |
| **超集**  | `a.issuperset(b)`           | `a >= b` | b 的所有元素都在 a      | -                        |
| **不相交** | `a.isdisjoint(b)`           | -        | 无公共元素            | -                        |

### (3)就地修改（更新集合）

| 操作 | 方法 | 运算符 | 效果 |
|------|------|--------|------|
| 并集更新 | `a.update(b)` | `a \|= b` | a = a ∪ b |
| 交集更新 | `a.intersection_update(b)` | `a &= b` | a = a ∩ b |
| 差集更新 | `a.difference_update(b)` | `a -= b` | a = a - b |
| 对称差更新 | `a.symmetric_difference_update(b)` | `a ^= b` | a = a △ b |

```python
a = {1, 2, 3}
a |= {3, 4, 5}      # a = {1,2,3,4,5}
a &= {2, 3, 4}      # a = {2,3,4}
```

### (4) 常用操作汇总

```python
s = {1, 2, 3, 4}

len(s)               # 4
x in s               # 判断存在
x not in s           # 判断不存在

# 浅拷贝
s.copy()             # 新集合
set(s)               # 另一种拷贝

# 删除所有元素
s.clear()
```

## 3.defaultdict`
`defaultdict` 是 `collections` 模块提供的字典子类，访问不存在的键时自动调用工厂函数生成默认值，避免 `KeyError`。

```python
from collections import defaultdict

d1 = defaultdict(int)      # 缺失键默认 0
d2 = defaultdict(list)     # 缺失键默认 []
d3 = defaultdict(set)      # 缺失键默认 set()
d4 = defaultdict(lambda: "默认值")

d1['a'] += 1               # 无需判断键是否存在
d2['b'].append(1)
```

## 4.deque
`deque`（双端队列）是 `collections` 模块提供的**两端高效操作**的队列，支持 O(1) 的左右增删。

```python
from collections import deque

dq = deque([1, 2, 3])
dq.append(4)        # 右端加 → [1,2,3,4]
dq.appendleft(0)    # 左端加 → [0,1,2,3,4]
dq.pop()            # 右端删 → 返回 4
dq.popleft()        # 左端删 → 返回 0
```

## 5.位运算

| 运算符  | 名称   | 示例                                |
| ---- | ---- | --------------------------------- |
| `&`  | 按位与  | `5 & 3` = `1` (101 & 011 = 001)   |
| `\|` | 按位或  | `5 \| 3` = `7` (101 \| 011 = 111) |
| `^`  | 按位异或 | `5 ^ 3` = `6` (101 ^ 011 = 110)   |
| `~`  | 按位取反 | `~5` = `-6`                       |
| `<<` | 左移   | `5 << 1` = `10` (101 → 1010)      |
| `>>` | 右移   | `5 >> 1` = `2` (101 → 10)         |

## 6.卡特兰数等

![](https://ik.imagekit.io/7ngad5bwp/homeworkpicture_0vQjWygd_)


## 7.几个概念

- **完全二叉树 (Complete Binary Tree)**：
    - 除了最后一层外，每一层都是满的。
    - 最后一层的节点都靠左排列。
- **正则二叉树 / 严格二叉树 (Full/Strict Binary Tree)**：
    - 每个节点要么没有子节点（叶子节点），要么有两个子节点。
- **满二叉树 (Perfect Binary Tree)**：
    - 所有层全满。

## 8.魔术方法

### （1） 对象生命周期
| 方法 | 作用 |
|------|------|
| `__new__(cls, ...)` | 创建并返回一个新实例（在 `__init__` 之前调用，常用于单例模式或不可变类型继承）。 |
| `__init__(self, ...)` | 初始化新创建的实例。 |
| `__del__(self)` | 析构器，对象被垃圾回收时调用（不保证立即执行）。 |

```python
class Demo:
    def __init__(self, value):
        self.value = value
        print(f"初始化: {value}")
```

### （2） 字符串表示
| 方法 | 作用 |
|------|------|
| `__repr__(self)` | 返回“官方”字符串表示，应尽量能通过 `eval` 还原对象，主要用于调试。 |
| `__str__(self)` | 返回“非正式”的用户友好字符串，`print()` 和 `str()` 使用。若未定义则回退到 `__repr__`。 |

```python
class Point:
    def __init__(self, x, y):
        self.x, self.y = x, y
    def __repr__(self):
        return f"Point({self.x}, {self.y})"
    def __str__(self):
        return f"({self.x}, {self.y})"

p = Point(1, 2)
print(repr(p))  # Point(1, 2)
print(str(p))   # (1, 2)
```

### （3） 算术运算符
二元运算符：`+` `-` `*` `/` `//` `%` `**` 等，都有对应魔术方法。  
如果左操作数不支持该运算，Python 会尝试右操作数的**反射方法**（带 `r` 前缀）。  
增强赋值（如 `+=`）对应**原地方法**（带 `i` 前缀）。

| 运算符 | 方法 | 反射方法 | 原地方法 |
|--------|------|----------|----------|
| `+` | `__add__` | `__radd__` | `__iadd__` |
| `-` | `__sub__` | `__rsub__` | `__isub__` |
| `*` | `__mul__` | `__rmul__` | `__imul__` |
| `/` | `__truediv__` | `__rtruediv__` | `__itruediv__` |
| `//` | `__floordiv__` | `__rfloordiv__` | `__ifloordiv__` |
| `%` | `__mod__` | `__rmod__` | `__imod__` |
| `**` | `__pow__` | `__rpow__` | `__ipow__` |

```python
class Vector:
    def __init__(self, x, y):
        self.x, self.y = x, y
    def __add__(self, other):
        return Vector(self.x + other.x, self.y + other.y)
    def __repr__(self):
        return f"Vector({self.x}, {self.y})"
```

### (4) 比较与哈希
| 方法                    | 作用                                                                 |
| --------------------- | ------------------------------------------------------------------ |
| `__eq__(self, other)` | ==                                                                 |
| `__ne__(self, other)` | `!=`（通常无需定义，会反用 `__eq__`）                                          |
| `__lt__(self, other)` | `<`                                                                |
| `__le__(self, other)` | `<=`                                                               |
| `__gt__(self, other)` | `>`                                                                |
| `__ge__(self, other)` | `>=`                                                               |
| `__hash__(self)`      | 返回哈希值，使对象可哈希（可作为字典键）。**定义了 `__eq__` 的对象默认不可哈希，需要同时定义 `__hash__`。** |
| `__bool__(self)`      | 用于 `bool(obj)`，若未定义则会尝试调用 `__len__`。                               |

```python
class User:
    def __init__(self, uid, name):
        self.uid = uid
        self.name = name
    def __eq__(self, other):
        return self.uid == other.uid
    def __hash__(self):
        return hash(self.uid)
```
### (5) 容器与迭代
| 方法 | 作用 |
|------|------|
| `__len__(self)` | `len(obj)` |
| `__getitem__(self, key)` | `obj[key]`（支持索引/切片） |
| `__setitem__(self, key, value)` | `obj[key] = value` |
| `__delitem__(self, key)` | `del obj[key]` |
| `__contains__(self, item)` | `item in obj` |
| `__iter__(self)` | 返回迭代器，使对象可迭代（`for` 循环使用） |
| `__next__(self)` | 迭代器的下一个值，抛出 `StopIteration` 结束迭代 |
| `__reversed__(self)` | `reversed(obj)` |

```python
class Countdown:
    def __init__(self, start):
        self.current = start
    def __iter__(self):
        return self
    def __next__(self):
        if self.current <= 0:
            raise StopIteration
        val = self.current
        self.current -= 1
        return val
```
