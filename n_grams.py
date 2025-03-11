import re
from collections import defaultdict, Counter
import random
import math
import argparse
import os
import pickle
from abc import ABC, abstractmethod

class BaseNgramModel(ABC):
    """Base class cho các mô hình N-gram"""
    
    def __init__(self, n):
        """
        Args:
            n (int): Độ dài của n-gram
        """
        self.n = n
        self.ngrams = defaultdict(Counter)
        self.context_totals = defaultdict(int)

    def tokenize(self, text):
        """Chuyển văn bản thành list các token
        Args:
            text (str): Văn bản đầu vào
        Returns:
            list: Danh sách các token
        """
        tokens = re.findall(r'\b\w+\b', text.lower())
        return tokens

    def train(self, text):
        """Huấn luyện mô hình trên một văn bản
        Args:
            text (str): Văn bản huấn luyện
        """
        tokens = self.tokenize(text)
        padded_tokens = ['<s>'] * (self.n - 1) + tokens + ['</s>']
        for i in range(len(tokens) + 1):
            context = tuple(padded_tokens[i:i + self.n - 1])
            word = padded_tokens[i + self.n - 1]
            self.ngrams[context][word] += 1
            self.context_totals[context] += 1

    @abstractmethod
    def predict(self, context):
        """Dự đoán xác suất của các từ tiếp theo
        Args:
            context (list): Ngữ cảnh
        Returns:
            dict: Xác suất của các từ tiếp theo
        """
        pass

    def generate(self, max_words=100):
        """Sinh văn bản
        Args:
            max_words (int): Số từ tối đa cần sinh
        Returns:
            str: Văn bản được sinh ra
        """
        context = ['<s>'] * (self.n - 1)
        result = []
        for _ in range(max_words):
            probabilities = self.predict(context)
            if probabilities is None:
                break
            words, probs = zip(*probabilities.items())
            next_word = random.choices(words, probs)[0]
            if next_word == '</s>':
                break
            result.append(next_word)
            context = context[1:] + [next_word]
        return ' '.join(result)

    def train_from_file(self, file_path):
        """Huấn luyện từ file
        Args:
            file_path (str): Đường dẫn đến file huấn luyện
        """
        texts = self.load_data(file_path)
        for text in texts:
            self.train(text)
        print(f"Đã huấn luyện xong với {len(texts)} câu từ {file_path}")

    def evaluate_file(self, file_path):
        """Đánh giá mô hình trên file
        Args:
            file_path (str): Đường dẫn đến file đánh giá
        Returns:
            dict: Kết quả đánh giá
        """
        texts = self.load_data(file_path)
        total_perplexity = 0
        total_log_likelihood = 0
        total_tokens = 0
        
        for text in texts:
            results = self.eval(text)
            total_perplexity += results['perplexity']
            total_log_likelihood += results['log_likelihood']
            total_tokens += results['total_tokens']
        
        n_texts = len(texts)
        return {
            'avg_perplexity': total_perplexity / n_texts,
            'total_log_likelihood': total_log_likelihood,
            'total_tokens': total_tokens,
            'n_texts': n_texts
        }

    def eval(self, test_text):
        """Đánh giá mô hình trên một văn bản
        Args:
            test_text (str): Văn bản test
        Returns:
            dict: Kết quả đánh giá
        """
        tokens = self.tokenize(test_text)
        padded_tokens = ['<s>'] * (self.n - 1) + tokens + ['</s>']
        log_likelihood = 0
        total_tokens = 0
        
        for i in range(len(tokens) + 1):
            context = tuple(padded_tokens[i:i + self.n - 1])
            word = padded_tokens[i + self.n - 1]
            probability = self.get_probability(context, word)
            
            if probability > 0:
                log_likelihood += math.log2(probability)
            else:
                log_likelihood += math.log2(1e-10)
            total_tokens += 1
        
        perplexity = 2 ** (-log_likelihood / total_tokens)
        return {
            'perplexity': perplexity,
            'log_likelihood': log_likelihood,
            'total_tokens': total_tokens
        }

    def inference(self, input_text, max_words=10):
        """Chạy inference trên câu đầu vào
        Args:
            input_text (str): Câu đầu vào
            max_words (int): Số từ tối đa cần sinh
        Returns:
            str: Kết quả sinh
        """
        tokens = self.tokenize(input_text)
        if len(tokens) < self.n - 1:
            tokens = ['<s>'] * (self.n - 1 - len(tokens)) + tokens
        
        context = tokens[-(self.n-1):]
        result = tokens.copy()
        
        for _ in range(max_words):
            probabilities = self.predict(context)
            if probabilities is None:
                break
            
            words, probs = zip(*probabilities.items())
            next_word = random.choices(words, probs)[0]
            
            if next_word == '</s>':
                break
                
            result.append(next_word)
            context = context[1:] + [next_word]
            
        return ' '.join(result)

    @abstractmethod
    def get_probability(self, context, word):
        """Lấy xác suất của một từ trong ngữ cảnh
        Args:
            context (tuple): Ngữ cảnh
            word (str): Từ cần tính xác suất
        Returns:
            float: Xác suất của từ
        """
        pass

    @staticmethod
    def load_data(file_path):
        """Đọc dữ liệu từ file
        Args:
            file_path (str): Đường dẫn đến file
        Returns:
            list: Danh sách các câu
        """
        texts = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                texts = [line.strip() for line in f if line.strip()]
        except Exception as e:
            print(f"Lỗi khi đọc file {file_path}: {str(e)}")
        return texts

    def save_model(self, file_path):
        """Lưu model xuống file
        Args:
            file_path (str): Đường dẫn file để lưu model
        """
        model_data = {
            'n': self.n,
            'ngrams': dict(self.ngrams),
            'context_totals': dict(self.context_totals)
        }
        try:
            with open(file_path, 'wb') as f:
                pickle.dump(model_data, f)
            print(f"Đã lưu model thành công vào {file_path}")
        except Exception as e:
            print(f"Lỗi khi lưu model: {str(e)}")

    @classmethod
    def load_model(cls, file_path):
        """Load model từ file
        Args:
            file_path (str): Đường dẫn đến file model
        Returns:
            BaseNgramModel: Instance của model đã load
        """
        try:
            with open(file_path, 'rb') as f:
                model_data = pickle.load(f)
            
            model = cls(model_data['n'])
            model.ngrams = defaultdict(Counter)
            model.context_totals = defaultdict(int)
            
            for context, counter in model_data['ngrams'].items():
                model.ngrams[context].update(counter)
            for context, total in model_data['context_totals'].items():
                model.context_totals[context] = total
                
            print(f"Đã load model thành công từ {file_path}")
            return model
        except Exception as e:
            print(f"Lỗi khi load model: {str(e)}")
            return None

class NgramModel(BaseNgramModel):
    """Mô hình N-gram cơ bản"""
    
    def predict(self, context):
        """Implement phương thức predict cho N-gram cơ bản"""
        if len(context) != self.n - 1:
            raise ValueError(f"Context phải có độ dài {self.n - 1}")
        context = tuple(context)
        if context in self.ngrams:
            total = self.context_totals[context]
            probabilities = {word: count / total 
                           for word, count in self.ngrams[context].items()}
            return probabilities
        return None

    def get_probability(self, context, word):
        """Implement phương thức get_probability cho N-gram cơ bản"""
        context = tuple(context)
        if context in self.ngrams and word in self.ngrams[context]:
            return self.ngrams[context][word] / self.context_totals[context]
        return 0

class BackoffNgramModel(BaseNgramModel):
    """Mô hình N-gram với kỹ thuật backoff"""
    
    def __init__(self, n, alpha=0.4):
        """
        Args:
            n (int): Độ dài của n-gram
            alpha (float): Hệ số giảm trọng số cho mỗi lần backoff
        """
        super().__init__(n)
        self.alpha = alpha
        self.lower_models = {
            i: NgramModel(i) for i in range(n-1, 0, -1)
        }

    def train(self, text):
        """Override phương thức train để huấn luyện cả các mô hình con"""
        super().train(text)
        for model in self.lower_models.values():
            model.train(text)

    def predict(self, context):
        """Implement phương thức predict với kỹ thuật backoff"""
        if len(context) != self.n - 1:
            raise ValueError(f"Context phải có độ dài {self.n - 1}")
            
        context = tuple(context)
        probs = super().predict(context)
        if probs is not None:
            return probs

        # Thực hiện backoff
        current_alpha = self.alpha
        for n in range(len(context)-1, -1, -1):
            shorter_context = context[-n:] if n > 0 else []
            model = self.lower_models[n+1]
            probs = model.predict(shorter_context)
            
            if probs is not None:
                return {word: prob * current_alpha 
                       for word, prob in probs.items()}
            current_alpha *= self.alpha

        return None

    def get_probability(self, context, word):
        """Implement phương thức get_probability với kỹ thuật backoff"""
        context = tuple(context)
        prob = super().get_probability(context, word)
        if prob > 0:
            return prob

        # Thực hiện backoff
        current_alpha = self.alpha
        for n in range(len(context)-1, -1, -1):
            shorter_context = context[-n:] if n > 0 else []
            model = self.lower_models[n+1]
            prob = model.get_probability(shorter_context, word)
            
            if prob > 0:
                return prob * current_alpha
            current_alpha *= self.alpha

        return 0

    def save_model(self, file_path):
        """Override phương thức save_model để lưu cả các mô hình con"""
        model_data = {
            'n': self.n,
            'alpha': self.alpha,
            'ngrams': dict(self.ngrams),
            'context_totals': dict(self.context_totals),
            'lower_models': {
                n: {
                    'ngrams': dict(model.ngrams),
                    'context_totals': dict(model.context_totals)
                }
                for n, model in self.lower_models.items()
            }
        }
        try:
            with open(file_path, 'wb') as f:
                pickle.dump(model_data, f)
            print(f"Đã lưu model backoff thành công vào {file_path}")
        except Exception as e:
            print(f"Lỗi khi lưu model backoff: {str(e)}")

    @classmethod
    def load_model(cls, file_path):
        """Override phương thức load_model để load cả các mô hình con"""
        try:
            with open(file_path, 'rb') as f:
                model_data = pickle.load(f)
            
            model = cls(model_data['n'], model_data['alpha'])
            model.ngrams = defaultdict(Counter)
            model.context_totals = defaultdict(int)
            
            # Load dữ liệu cho mô hình chính
            for context, counter in model_data['ngrams'].items():
                model.ngrams[context].update(counter)
            for context, total in model_data['context_totals'].items():
                model.context_totals[context] = total
            
            # Load dữ liệu cho các mô hình con
            for n, lower_data in model_data['lower_models'].items():
                model.lower_models[n] = NgramModel(n)
                for context, counter in lower_data['ngrams'].items():
                    model.lower_models[n].ngrams[context].update(counter)
                for context, total in lower_data['context_totals'].items():
                    model.lower_models[n].context_totals[context] = total
                    
            print(f"Đã load model backoff thành công từ {file_path}")
            return model
        except Exception as e:
            print(f"Lỗi khi load model backoff: {str(e)}")
            return None

def parse_args():
    """
    Xử lý command line arguments
    """
    parser = argparse.ArgumentParser(description='Huấn luyện và inference mô hình N-gram')
    
    parser.add_argument('--mode', type=str, required=True, 
                       choices=['train', 'inference'],
                       help='Chế độ chạy: train (huấn luyện, đánh giá và lưu model) hoặc inference (load model và dự đoán)')
    
    parser.add_argument('--model_type', type=str, default='normal',
                       choices=['normal', 'backoff'],
                       help='Loại mô hình: normal (N-gram thường) hoặc backoff (N-gram với backoff)')
    
    parser.add_argument('--n', type=int, default=3,
                       help='Độ dài của n-gram (mặc định: 3)')
    
    parser.add_argument('--train_file', type=str,
                       help='File huấn luyện (required cho mode train)')
    
    parser.add_argument('--valid_file', type=str,
                       help='File validation (required cho mode train)')
    
    parser.add_argument('--model_path', type=str, required=True,
                       help='Đường dẫn để lưu/load model')
    
    parser.add_argument('--input_text', type=str,
                       help='Câu đầu vào cho inference (required cho mode inference)')
    
    parser.add_argument('--max_words', type=int, default=10,
                       help='Số từ tối đa cần sinh trong inference (mặc định: 10)')
    
    parser.add_argument('--alpha', type=float, default=0.4,
                       help='Hệ số alpha cho mô hình backoff (mặc định: 0.4)')
    
    args = parser.parse_args()
    
    # Kiểm tra điều kiện
    if args.mode == 'train' and (not args.train_file or not args.valid_file):
        parser.error("Chế độ train yêu cầu cả --train_file và --valid_file")
    
    if args.mode == 'inference' and not args.input_text:
        parser.error("Chế độ inference yêu cầu --input_text")
    
    return args

def main():
    args = parse_args()

    if args.mode == 'train':
        # Khởi tạo model mới
        if args.model_type == 'normal':
            model = NgramModel(n=args.n)
        else:
            model = BackoffNgramModel(n=args.n, alpha=args.alpha)

        # Huấn luyện
        print(f"Đang huấn luyện mô hình từ file {args.train_file}...")
        model.train_from_file(args.train_file)

        # Đánh giá
        print("\nĐang đánh giá mô hình...")
        eval_results = model.evaluate_file(args.valid_file)
        print(f"Kết quả đánh giá:")
        print(f"Perplexity trung bình: {eval_results['avg_perplexity']:.2f}")
        print(f"Tổng log likelihood: {eval_results['total_log_likelihood']:.2f}")
        print(f"Tổng số token: {eval_results['total_tokens']}")
        print(f"Số câu đánh giá: {eval_results['n_texts']}")

        # Lưu model
        model.save_model(args.model_path)
        print(f"Đã lưu model vào {args.model_path}")

    else:  # inference mode
        # Load model
        ModelClass = BackoffNgramModel if args.model_type == 'backoff' else NgramModel
        model = ModelClass.load_model(args.model_path)
        if model is None:
            return

        # Chạy inference
        print("\nKết quả inference:")
        print(f"Đầu vào: {args.input_text}")
        result = model.inference(args.input_text, max_words=args.max_words)
        print(f"Kết quả: {result}")

if __name__ == "__main__":
    main()
