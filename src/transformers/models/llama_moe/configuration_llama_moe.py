from transformers.configuration_utils import PretrainedConfig


class LlamaMoEConfig(PretrainedConfig):
    model_type = "llama_moe"

    def __init__(
        self,
        vocab_size=32000,
        hidden_size=2048,
        intermediate_size=5504,
        num_hidden_layers=26,
        num_attention_heads=16,
        num_key_value_heads=16,
        hidden_act="silu",
        max_position_embeddings=4096,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        pad_token_id=None,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=False,

        # ----- MoE config -----
        num_experts=16,
        num_selects=2,
        gate_type="TopKBalancedNoisyGate",
        gate_network="mlp",
        gate_use_softmax=True,
        gate_use_balance=True,
        gate_balance_loss_weight=1e-2,
        gate_add_noise=True,
        gate_noise_epsilon=1e-2,

        calculator_type="UniversalCalculator",
        drop_tokens=True,
        capacity_factor=1.25,
        min_capacity=4,

        # >>> YOUR NEW FIELD <<<
        routing_policy="topk_noisy",

        **kwargs,
    ):
        super().__init__(**kwargs)

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.tie_word_embeddings = tie_word_embeddings
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id

        # MoE parameters
        self.num_experts = num_experts
        self.num_selects = num_selects
        self.gate_type = gate_type
        self.gate_network = gate_network
        self.gate_use_softmax = gate_use_softmax
        self.gate_use_balance = gate_use_balance
        self.gate_balance_loss_weight = gate_balance_loss_weight
        self.gate_add_noise = gate_add_noise
        self.gate_noise_epsilon = gate_noise_epsilon

        self.calculator_type = calculator_type
        self.drop_tokens = drop_tokens
        self.capacity_factor = capacity_factor
        self.min_capacity = min_capacity

        # >>> Your new field assignment <<<
        self.routing_policy = routing_policy
