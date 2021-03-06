import cupy as np


class AnomaryDetector:
    err_for_normal_set = None
    vec_dim = 0
    mu = 0
    inv_sigma = 0

    def __init__(self, net, seq_length, dim, calc_length=None):
        """
        :param net: chainerにて実装+学習済みのEndDec_ADモデル
        :param seq_length: 入力する時系列データの時系列長
        :param dim: 時系列データの各時刻のデータベクトルの次元
        :param calc_length: 時系列データの内、異常検知に利用する時刻の範囲 先頭からどの時刻までか指定
        """
        self.net = net
        self.seq_length = seq_length
        self.vec_dim = dim

        if calc_length is None:
            self.calc_length = seq_length
        elif seq_length >= calc_length:
            self.calc_length = calc_length
        else:
            raise ValueError("calc_length must be smaller than seq_length")

    def fit(self, normal_seq):
        """
        時系列データに対し,self.netでの再構成誤差ベクトルを計算し,多変量正規分布にフィッティングします
        :param normal_seq: 正常時系列データ
        :return: なし
        """
        if self.err_for_normal_set is None:  # errが空リストだったら新規作成
            self.err_for_normal_set = self.calc_err_vec(normal_seq)
        else:  # errがndarrayで中身が入っていれば結合
            self.err_for_normal_set = np.concatenate([self.err_for_normal_set, self.calc_err_vec(normal_seq)])

    def fit2(self):
        # 平均,不偏共分散行列の逆行列を計算,更新
        self.mu = np.mean(self.err_for_normal_set, axis=0)
        sigma = np.cov(self.err_for_normal_set, rowvar=False)
        self.inv_sigma = np.linalg.inv(sigma)

    def reconstruct(self, origin_seq):
        """
        :param origin_seq: array-like
            時系列データ [[t_0], [t_1], ..., [t_l-1]] = [[array of float], [array of float], ...]
        :return: [variable([[skelton_pos_t0]]), variable([[skelton_pos_t1]]), ...]
        """
        return self.net(origin_seq)

    def calc_anomary_score(self, anomalous_seq):
        """
        :param anomalous_seq:異常度を算出したいデータ, ベクトル
        :return: anomary score
        """
        anomary_score = []
        err = self.calc_err_vec(anomalous_seq)

        for e in err[:self.calc_length]:
            score = np.dot(np.dot(e - self.mu, self.inv_sigma), e - self.mu)
            anomary_score.append(score)

        return anomary_score

    def calc_err_vec(self, raw_seq):
        """
        与えた時系列データに対し再構成を行い, 誤差ベクトルを計算します
        :param raw_seq:
        :return:
        """
        err = []

        reconst_seq = self.reconstruct(raw_seq)[0].array  # cupy
        raw_seq = raw_seq[0]  # cupy

        for t in range(self.calc_length):
            err.append(np.abs(raw_seq[t] - reconst_seq[t]))

        return np.array(err)


    def reset(self):
        self.err_for_normal_set = None
        self.vec_dim = 0
        self.mu = 0
        self.inv_sigma = 0
