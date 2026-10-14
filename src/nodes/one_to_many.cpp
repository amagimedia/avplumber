#include "one_to_many.hpp"

// On a thread of its own, sleeping until the input has a frame.
template <typename T>
class OneToMany : public OneToManyBase<T> {
public:
    using OneToManyBase<T>::OneToManyBase;

    virtual void process() override {
        T* data = this->source_->peek();
        if (!data) return;
        this->handOut(data);
        this->source_->pop();
    }

    static std::shared_ptr<OneToMany> create(NodeCreationInfo& nci) {
        return OneToManyBase<T>::template build<OneToMany>(nci);
    }
};

DECLNODE_ATD(one_to_many, OneToMany);
